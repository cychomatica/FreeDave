'''
adpated from: 
    https://github.com/Gen-Verse/dLLM-RL
'''
import torch
import math
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Optional, List, Union


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x

@torch.no_grad()
def block_diffusion_generate_FreeDave(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        draft_steps=4,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    # block_priority_shift = torch.cat([torch.zeros(prompt.shape[1], device=x.device), 
                                    # torch.arange(num_blocks-1, -1, -1, device=x.device).repeat_interleave(block_length)])

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]

        # First forward pass
        step = 0
        x0, x0_p = sample_step(model, cur_x, cur_attn_mask, cur_position_ids, past_key_values, temperature, top_k, top_p)
    
        while step < denoising_steps:

            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                break
            
            # Draft token filling from last forward pass
            cur_draft_steps = min(draft_steps, denoising_steps - step)
            # if cur_draft_steps == 1:
            #     cur_x = token_filling(
            #         cur_x, 
            #         x0, 
            #         x0_p, 
            #         mask_id, 
            #         remasking_strategy, 
            #         confidence_threshold, 
            #         num_transfer_tokens, 
            #         [step], 
            #         cur_draft_steps)
            #     break
            
            x_draft = token_filling(
                cur_x, 
                x0, 
                x0_p, 
                mask_id, 
                remasking_strategy, 
                confidence_threshold, 
                num_transfer_tokens, 
                [step], 
                cur_draft_steps
            )

            if denoising_steps - step > 1:
                past_key_values.batch_repeat_interleave(x_draft.shape[0])

                # New forward pass
                x0_draft, x0_p_draft = sample_step(model, x_draft.clone(), cur_attn_mask, cur_position_ids, past_key_values, temperature, top_k, top_p)

                # Target token filling from new forward pass
                x_target = token_filling(
                    x_draft.clone(), 
                    x0_draft, 
                    x0_p_draft, 
                    mask_id, 
                    remasking_strategy, 
                    confidence_threshold, 
                    num_transfer_tokens, 
                    [step + i + 1 for i in range(cur_draft_steps)], 
                    1
                )

                # Draft tokens verification
                x_draft = x_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:]) # [batch, cur_draft_steps, seq_len]
                x_target = x_target.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:]) # [batch, cur_draft_steps, seq_len]
                matched_draft_index = (x_target[:, :-1, :] == x_draft[:, 1:, :]).all(dim=-1)
                matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1) #NOTE: assert matched_steps.shape[0] == 1 for now

                # Update current x, current kv cache, and forward pass intermediate results
                cur_x = x_draft[:, matched_steps, :].squeeze(1)
                x0 = x0_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, matched_steps, :].squeeze(1)
                x0_p = x0_p_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, matched_steps, :].squeeze(1)
                past_key_values.batch_select_indices(matched_steps)

                # Update step
                step += matched_steps + 1
            else:
                cur_x = x_draft
                step += 1

        # Update kv cache
        if num_block < num_blocks - 1:
            model(cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True)

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x

@torch.no_grad()
def block_diffusion_generate_FreeDave_v1(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        draft_steps=4,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # TODO: dynamic block

    # Decode stage
    # gen_blocks = 0
    # while gen_blocks < num_blocks - prefill_blocks:
    # for num_block in range(num_blocks - prefill_blocks):
    for num_block in range(prefill_blocks, num_blocks):

        # NOTE: assume block_length == denoising_steps for now
        cur_num_draft_blocks = min(math.ceil(draft_steps / denoising_steps), num_blocks - num_block - 1)
        cur_num_transfer_tokens = num_transfer_tokens.repeat(cur_num_draft_blocks+1)

        cur_x = x[:, prefill_length:prefill_length + (cur_num_draft_blocks+1) * block_length].clone()
        cur_attn_mask = torch.ones(x.shape[0], (cur_num_draft_blocks+1) * block_length, prefill_length + (cur_num_draft_blocks+1) * block_length, device=x.device) # each token in current block and draft blocks attends to all previous blocks, current block, and draft blocks
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, prefill_length:prefill_length + (cur_num_draft_blocks+1) * block_length]
        block_priority_shift = torch.arange(cur_num_draft_blocks, -1, -1, device=x.device).repeat_interleave(block_length)

        # First forward pass
        # step = 0
        step = (cur_x != mask_id).sum()
        if step < denoising_steps:
            x0, x0_p = sample_step(model, cur_x, cur_attn_mask, cur_position_ids, past_key_values, temperature, top_k, top_p)
            while step < denoising_steps:

                mask_index = (cur_x == mask_id)
                if mask_index[:, :block_length].sum() == 0:
                    break
                
                # Draft token filling from last forward pass
                if num_block < num_blocks - 1:
                    cur_draft_steps = min(draft_steps, denoising_steps * cur_num_draft_blocks)
                else:
                    cur_draft_steps = min(draft_steps, denoising_steps - step)

                if step == denoising_steps - 1 and num_block == num_blocks - 1: # if last block and last step, exit
                    cur_x = token_filling(
                        cur_x, 
                        x0, 
                        x0_p, 
                        mask_id, 
                        remasking_strategy, 
                        confidence_threshold, 
                        cur_num_transfer_tokens, 
                        [step], 
                        cur_draft_steps, block_priority_shift
                    )
                    break
                
                x_draft = token_filling(
                    cur_x, 
                    x0, 
                    x0_p, 
                    mask_id, 
                    remasking_strategy, 
                    confidence_threshold, 
                    cur_num_transfer_tokens, 
                    [step], 
                    cur_draft_steps, 
                    block_priority_shift
                )

                # if denoising_steps - step > 1:
                past_key_values.batch_repeat_interleave(x_draft.shape[0])

                # New forward pass
                x0_draft, x0_p_draft = sample_step(model, x_draft.clone(), cur_attn_mask, cur_position_ids, past_key_values, temperature, top_k, top_p)

                # Target token filling from new forward pass
                x_target = token_filling(
                    x_draft.clone(), 
                    x0_draft, 
                    x0_p_draft, 
                    mask_id, 
                    remasking_strategy, 
                    confidence_threshold, 
                    cur_num_transfer_tokens, 
                    [step + i + 1 for i in range(cur_draft_steps)], 1, block_priority_shift
                )

                # Draft tokens verification
                x_draft = x_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:]) # [batch, cur_draft_steps, seq_len]
                x_target = x_target.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:]) # [batch, cur_draft_steps, seq_len]
                matched_draft_index = (x_target[:, :-1, :] == x_draft[:, 1:, :]).all(dim=-1)
                matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1) #NOTE: assert matched_steps.shape[0] == 1 for now

                # Update current x, current kv cache, and forward pass intermediate results
                cur_x = x_draft[:, matched_steps, :].squeeze(1)
                x0 = x0_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, matched_steps, :].squeeze(1)
                x0_p = x0_p_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, matched_steps, :].squeeze(1)
                past_key_values.batch_select_indices(matched_steps)
                
                # Update step
                step += matched_steps.item() + 1
    
                # NOTE: more aggressive matching (not ready yet)
                # region
                # matched_select_index = (x_target[:, :-1, :] == x_draft[:, 1:, :]).prod(dim=1).bool()
                # cur_x[matched_select_index] = x_draft[:, -1, :][matched_select_index]
                # x0[matched_select_index] = x0_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, -1, :][matched_select_index]
                # x0_p[matched_select_index] = x0_p_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])[:, -1, :][matched_select_index]
                # past_key_values.batch_select_indices(torch.tensor([-1], device=x.device))
                # step += matched_select_index.sum() + 1
                # endregion

        # Update kv cache
        if num_block < num_blocks - 1: # if not the last block
            model(cur_x[:, :block_length],
                    attention_mask=cur_attn_mask[:, :, :block_length, :prefill_length+block_length],
                    position_ids=cur_position_ids[:, :block_length],
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True)

        # Fill current block; update prefill_length and num_block
        x[:, prefill_length:prefill_length + (cur_num_draft_blocks+1) * block_length] = cur_x
        prefill_length += block_length

        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x

@torch.no_grad()
def sample_step(model, 
                x, 
                attention_mask, 
                position_ids, 
                past_key_values, 
                temperature, 
                top_k, 
                top_p, 
                ):

    # Denosing
    logits = model(x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False).logits

    # Sampling
    x0, x0_p = sample_with_temperature_topk_topp(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return x0, x0_p

@torch.no_grad()
def token_filling(
    x: torch.Tensor, 
    x0: torch.Tensor, 
    x0_p: torch.Tensor, 
    mask_id: int, 
    remasking_strategy: str, 
    confidence_threshold: float, 
    num_transfer_tokens: torch.Tensor, 
    step: Union[torch.Tensor, List[int]], 
    draft_steps: int=1, 
    block_priority_shift: Optional[torch.Tensor] = None
):

    x_draft = x.unsqueeze(1).expand(x.shape[0], draft_steps, *x.shape[1:]).clone() # (batch, draft_steps, seq_len)
    mask_index = (x == mask_id)
    mask_index_draft = (x_draft == mask_id)

    # Remasking strategy
    if remasking_strategy == 'sequential':
        transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)
        for j in range(x_draft.shape[0]):
            for k in range(draft_steps):
                if mask_index_draft[j, k].any():
                    first_mask_index = mask_index_draft[j, k].nonzero(as_tuple=True)[
                        0].min().item()
                    transfer_index[j, k, first_mask_index:first_mask_index + num_transfer_tokens[step: step + k + 1].sum()] = True
                else:
                    raise ValueError(
                        "No mask tokens found in the current block.")

    elif remasking_strategy == 'low_confidence_static':
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        if block_priority_shift is not None:
            confidence = confidence + block_priority_shift
        transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            for k in range(draft_steps):
                _, idx = torch.topk(confidence[j], num_transfer_tokens[step[j]: step[j] + k + 1].sum())
                transfer_index[j, k, idx] = True

    #TODO: adaptation
    # elif remasking_strategy == 'low_confidence_dynamic':
    #     confidence = torch.where(mask_index, x0_p, -torch.inf)
    #     transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)
    #     for j in range(confidence.shape[0]):
    #         high_conf_mask = confidence[j] > confidence_threshold
    #         num_high_confidence = high_conf_mask.sum()

    #         for k in range(draft_steps):
    #             if num_high_confidence >= num_transfer_tokens[step: step + k + 1].sum():
    #                 transfer_index[j, k] = high_conf_mask
    #             else:
    #                 _, idx = torch.topk(confidence[j], num_transfer_tokens[step: step + k + 1].sum())
    #                 transfer_index[j, k, idx] = True

    else:
        raise ValueError(
            f"Unknown remasking strategy: {remasking_strategy}")

    x_draft[transfer_index] = x0.unsqueeze(1).expand_as(x_draft)[transfer_index]
    x_draft = x_draft.view(x.shape[0] * draft_steps, *x.shape[1:]) # (batch * draft_steps, seq_len)

    return x_draft