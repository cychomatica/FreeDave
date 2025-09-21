from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
from modeling.llada.modeling_llada import LLaDAModelLM
from typing import Optional

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@dataclass
class DiffusionOutput:
    sequences: torch.Tensor               # final result  (B, L_total)  (GPU)
    history:   List[torch.Tensor]         # all intermediate x (CPU)
    nfe:       int

@torch.no_grad()
def generate_with_prefix_cache(
        model, 
        input_ids,
        steps, 
        gen_length, 
        block_length, 
        temperature, 
        strategy, 
        mask_id, 
        further_horizon, 
        use_cache, 
        unmask_threshold
    ) -> DiffusionOutput:

    cgws = further_horizon
    histories: List[torch.Tensor] = []
    nfe = 0

    x = F.pad(input_ids, (0, gen_length), value=mask_id)
    max_length = input_ids.shape[1] + gen_length

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]    

    for num_block in range(num_blocks):
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        if cgws is not None:
            window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)
        
        cur_denoising_steps = steps_per_block[num_block]
        cur_num_transfer = get_num_transfer_tokens((x[:, current_block_start:current_block_end] == mask_id), cur_denoising_steps)

        # first full forward to build prefix cache
        if use_cache:
            model_output = model(x, use_cache=True)
            past_key_values = model_output.past_key_values
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
            past_key_values = new_past_key_values
        else:
            model_output = model(x, use_cache=False)
        
        mask_index_all = (x == mask_id)
        mask_index_all[:, current_block_end:] = 0

        x0, transfer_index = get_transfer_index(
            model_output.logits, temperature, strategy, mask_index_all,
            x, cur_num_transfer[:, 0], unmask_threshold)
        x[transfer_index] = x0[transfer_index]
        histories.append(x.clone().cpu())
        nfe += 1

        i = 1
        while True:
            nfe += 1
            if cgws is not None:
                mask_index_block = (x[:, window_slice] == mask_id)
            else:
                mask_index_block = (x[:, current_block_start:] == mask_id)
            mask_index_block[:, block_length:] = 0

            if use_cache:
                if cgws is not None:
                    logits = model(x[:, window_slice], past_key_values=past_key_values, use_cache=True).logits
                    x0, transfer_index = get_transfer_index(
                        logits, temperature, strategy,
                        mask_index_block, x[:, window_slice], cur_num_transfer[:, i], unmask_threshold)
                    x[:, window_slice][transfer_index] = x0[transfer_index]
                else:
                    logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
                    x0, transfer_index = get_transfer_index(
                        logits, temperature, strategy,
                        mask_index_block, x[:, current_block_start:], cur_num_transfer[:, i], unmask_threshold)
                    x[:, current_block_start:][transfer_index] = x0[transfer_index]
            else:
                logits = model(x, use_cache=False).logits
                logits = logits[:, current_block_start:]
                x0, transfer_index = get_transfer_index(
                    logits, temperature, strategy,
                    mask_index_block, x[:, current_block_start:], cur_num_transfer[:, i], unmask_threshold)
                x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            histories.append(x.clone().cpu())

            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    return DiffusionOutput(sequences=x, history=histories, nfe=nfe)

def get_transfer_index(logits, temperature, strategy, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if strategy == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif strategy == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values            # (b, l, 2)
        x0_p = top2[..., 0] - top2[..., 1]                # Δ(top1, top2)
    elif strategy == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # –entropy
    elif strategy == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(strategy)
    
    x0 = torch.where(mask_index, x0, x)
    
    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)  # (B, T)

        has_mask = mask_index.any(dim=-1)               # (B,)
        none_sel = (~selected.any(dim=-1)) & has_mask   # (B,)
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)     # (B,)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True

        return x0, selected

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index

@torch.no_grad()
def generate_with_prefix_cache_FreeDave(
        model, 
        input_ids,
        steps, 
        draft_steps,
        gen_length, 
        block_length, 
        temperature, 
        strategy, 
        mask_id, 
        further_horizon, 
        use_cache, 
        unmask_threshold
    ) -> DiffusionOutput:

    cgws = further_horizon
    histories: List[torch.Tensor] = []
    nfe = 0

    x = F.pad(input_ids, (0, gen_length), value=mask_id)
    max_length = input_ids.shape[1] + gen_length

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]    

    for num_block in range(num_blocks):
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        if use_cache:
            if cgws is not None:
                window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
                window_slice = slice(current_block_start, window_end)
                cur_x = x[:, window_slice]
            else:
                cur_x = x[:, current_block_start:]
        else:
            cur_x = x

        cur_denoising_steps = steps_per_block[num_block]
        cur_num_transfer = get_num_transfer_tokens((x[:, current_block_start:current_block_end] == mask_id), cur_denoising_steps)

        # update cache
        if use_cache:
            model_output = model(x, use_cache=True)
            past_key_values = model_output.past_key_values
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
            past_key_values = new_past_key_values
        else:
            model_output = model(x, use_cache=False)
        
        mask_index_all = (x == mask_id)
        mask_index_all[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(
            model_output.logits, temperature, strategy, mask_index_all,
            x, cur_num_transfer[:, 0], unmask_threshold)
        x[transfer_index] = x0[transfer_index]
        histories.append(x.clone().cpu())
        nfe += 1

        # first step denoising for draft token sampling
        mask_index_block = (cur_x == mask_id)
        mask_index_block[:, block_length:] = False
        x0, x0_p = sample_step(
            model, cur_x, current_block_start, past_key_values, use_cache, temperature, strategy
        )
        nfe += 1

        step = 1
        while step < cur_denoising_steps:

            cur_draft_steps = min(draft_steps, cur_denoising_steps - step)
            if (cur_x[:, :block_length] == mask_id).sum() == 0:
                break

            x_draft = token_filling(
                cur_x, 
                x0, 
                x0_p, 
                [step], 
                cur_draft_steps,
                cur_num_transfer, 
                mask_index_block,
                unmask_threshold
            )
            
            if cur_denoising_steps - step > 1:
                if use_cache:
                    past_key_values = cache_batch_repeat_interleave(past_key_values, cur_draft_steps)
                mask_index_draft = (x_draft == mask_id)
                mask_index_draft[:, block_length:] = False
                x0_draft, x0_p_draft = sample_step(
                    model, x_draft, current_block_start, past_key_values, use_cache, temperature, strategy
                )

                x_target = token_filling(
                    x_draft, 
                    x0_draft, 
                    x0_p_draft, 
                    [step + i + 1 for i in range(cur_draft_steps)], 
                    1,
                    cur_num_transfer.expand(x_draft.shape[0], *cur_num_transfer.shape[1:]), 
                    mask_index_draft,
                    unmask_threshold
                )
                
                x_draft = x_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])
                x_target = x_target.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])
                matched_draft_index = (x_target[:, :-1, :] == x_draft[:, 1:, :]).all(dim=-1)
                matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1) #NOTE: assert matched_steps.shape[0] == 1 for now
                # matched_steps.zero_() # debug
                cur_x = x_draft[:, matched_steps, :].squeeze(1)
                x0 = x0_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, matched_steps, :].squeeze(1)
                x0_p = x0_p_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, matched_steps, :].squeeze(1)
                past_key_values = cache_batch_select_indices(past_key_values, matched_steps)
                step += matched_steps.item() + 1
            else:
                cur_x = x_draft
                step += 1

            if cgws is not None:
                x[:, window_slice] = cur_x
            else:
                x[:, current_block_start:] = cur_x

            if histories is not None:
                histories.append(x.clone().cpu())

            mask_index_block = (cur_x == mask_id)
            mask_index_block[:, block_length:] = False
            
            if (cur_x[:, :block_length] == mask_id).sum() == 0:
                break

    return DiffusionOutput(sequences=x, history=histories, nfe=nfe)

@torch.no_grad()
def sample_tokens(logits, temperature, target="confidence"):

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values            # (b, l, 2)
        x0_p = top2[..., 0] - top2[..., 1]                # Δ(top1, top2)
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # –entropy
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(target)
    
    return x0, x0_p

@torch.no_grad()
def sample_step(
    model, 
    x, 
    current_block_start, 
    past_key_values,
    use_cache, 
    temperature, 
    target):

    if use_cache:
        logits = model(x, past_key_values=past_key_values, use_cache=True).logits
    else:
        logits = model(x, use_cache=False).logits
        logits = logits[:, current_block_start:]

    x0, x0_p = sample_tokens(logits, temperature, target)

    return x0, x0_p

@torch.no_grad()
def token_filling(
    x: torch.Tensor,
    x0: torch.Tensor,
    x0_p: torch.Tensor, 
    step: Optional[torch.Tensor, List[int]], 
    draft_steps: int, 
    num_transfer_tokens: torch.Tensor, 
    mask_index: torch.Tensor, 
    unmask_threshold: Optional[float] = None
):
    x_draft = x.unsqueeze(1).expand(x.shape[0], draft_steps, *x.shape[1:]).clone()
    
    if unmask_threshold is None:
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)
        for j in range(x_draft.shape[0]):
            for k in range(draft_steps):
                _, idx = torch.topk(confidence[j], num_transfer_tokens[j, step[j]: step[j] + k + 1].sum())
                transfer_index[j, k, idx] = True
    # else:
    #     selected = mask_index & (x0_p >= unmask_threshold)  # (B, T)

    #     has_mask = mask_index.any(dim=-1)               # (B,)
    #     none_sel = (~selected.any(dim=-1)) & has_mask   # (B,)
    #     if none_sel.any():
    #         masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
    #         best_idx = masked_scores.argmax(dim=-1)     # (B,)
    #         rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
    #         selected[rows, best_idx[rows]] = True

    x_draft[transfer_index] = x0.unsqueeze(1).expand_as(x_draft)[transfer_index]
    x_draft = x_draft.view(x.shape[0] * draft_steps, *x.shape[1:]) # (batch * draft_steps, seq_len)
    return x_draft


    #     return x0, selected

    # return x0, transfer_index

    
    # for k in range(draft_steps):
    #     x0_, transfer_index_ = get_transfer_index(
    #                                     logits, temperature, strategy,
    #                                     mask_index, x, num_transfer_tokens[:, step: step+k+1].sum(dim=-1), unmask_threshold
    #                                     )
    #     x0[:, k, :] = x0_
    #     transfer_index[:, k, :] = transfer_index_

@torch.no_grad()
def cache_batch_repeat_interleave(past_key_values, n):
    '''
        past_key_values: 
        a tuple of n_heads tuples (key, value)
        key: (batch, seq_len, head_dim)
        value: (batch, seq_len, head_dim)

        new_past_key_values:
        a list of n_heads tuples (key, value)
        key: (batch * n, seq_len, head_dim)
        value: (batch * n, seq_len, head_dim)
    '''

    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            new_past_key_values[i] += (past_key_values[i][j].expand(n, *past_key_values[i][j].shape[1:]),)
    return new_past_key_values

@torch.no_grad()
def cache_batch_select_indices(past_key_values, indices):
    '''
        past_key_values: 
        a tuple of n_heads tuples (key, value)
        key: (batch, seq_len, head_dim)
        value: (batch, seq_len, head_dim)
    '''
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            new_past_key_values[i] += (past_key_values[i][j][indices, ...],)
    return new_past_key_values