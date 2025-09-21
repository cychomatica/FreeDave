from __future__ import annotations
import math, json, os, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F

from modeling.dream import DreamTokenizer
from modeling.dream.modeling_dream import DreamModel
from modeling.dream.generation_utils_block import DreamGenerationMixin
import types
from modeling.dream.generation_utils_block import DreamGenerationConfig


from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
import torch.distributions as dists
from dataclasses import dataclass
from torch.nn import functional as F
import torch


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, tar=None):

    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs

    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)
    
    if tar == "confidence":
        return target, x0
    
    if tar == "margin_confidence":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        target = top1_probs - top2_probs 
    
    if tar == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        target = torch.sum(probs * log_probs, dim=-1)
    
    return target, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


@torch.no_grad()
def block_diffusion_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: Optional[float] = 0.9
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    gen_length = generation_config.max_gen_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, gen_length), value=mask_token_id)
    max_length = gen_length + input_ids.shape[1]
    
    # Handle block configuration
    if block_length is None:
        block_length = gen_length  # Default: single block (original behavior)
    
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"

    # Initialize cache for the prompt
    past_key_values = None

    # Process each block
    for num_block in range(num_blocks):
        
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        if cgws is not None:
            window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)

        # update cache
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            # Extract only previous block cache
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values

        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)
        
        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, current_block_start] = x0[:, current_block_start]
        if histories is not None:
            histories.append(x.clone().cpu())
        
        
        spb = steps_per_block[num_block]
        i = 1
        while True:
            
            
            if cgws is not None:
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)
            
            
            # Prepare attention mask for cached generation
            if attention_mask != "full":
                # Adjust attention mask for current position
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask
            
            if use_cache:
                if cgws is not None:
                    model_output = model(x[:, window_slice], current_attention_mask, 
                                    tok_idx[:, window_slice] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                else:
                    model_output = model(x[:, current_block_start:], current_attention_mask,
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits
                logits = logits[:, current_block_start:]
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
            
            
            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            target, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, tar=tar)

            # —— pad token penalty ——
            _pad_target_divisor = pad_target_penalty
            _pad_mask_flat = (x0 == pad_token_id)  
            if _pad_mask_flat.any():
                target = target.clone()
                target[_pad_mask_flat] = target[_pad_mask_flat] / _pad_target_divisor

            if cgws is not None:
                full_target = torch.full_like(x[:, window_slice], -torch.inf, device=model.device, dtype=logits.dtype)
            else:
                full_target = torch.full_like(x[:, current_block_start:], -torch.inf, device=model.device, dtype=logits.dtype)
            full_target = full_target.float()
            full_target[mask_index] = target
            full_target[:, block_length:] = -torch.inf

            if unmask_threshold is None:

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                t = timesteps[num_block][i]
                s = timesteps[num_block][i + 1]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < spb - 1 else int(num_mask_token)
                
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                    else:
                        full_target = full_target / alg_temp
                        full_target = F.softmax(full_target, dim=-1)
                        transfer_index = torch.multinomial(full_target, num_samples=number_transfer_tokens)
                    
                    if cgws is not None:
                        x_ = torch.zeros_like(x[:, window_slice], device=model.device, dtype=torch.long) + mask_token_id
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=model.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)

                    
                    
                    if cgws is not None:
                        x[:, window_slice][row_indices,transfer_index] = x_[row_indices,transfer_index]
                    else:
                        x[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]
                    
                
                    
            else:
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, current_block_start:]
                
                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
                selected_map[mask_index] = (target >= unmask_threshold)
                no_sel = ~selected_map.any(dim=-1)  # [B]
                no_sel = no_sel & mask_index.any(dim=-1)

                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True

                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

                xwin[selected_map] = x_candidates[selected_map]

                
            
            if histories is not None:
                histories.append(x.clone().cpu())

            i += 1

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
        
        # block_all_pad = torch.all(
        #     x[:, current_block_start:current_block_end] == pad_token_id
        # )
        # if block_all_pad:
        #     if current_block_end < x.size(1):
        #         x[:, current_block_end:] = pad_token_id
        #     if histories is not None:
        #         histories.append(x.clone().cpu())
        #     break

    
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x


@torch.no_grad()
def block_diffusion_generate_(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: Optional[float] = 0.9
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    gen_length = generation_config.max_gen_length
    steps = generation_config.steps
    draft_steps = generation_config.draft_steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, gen_length), value=mask_token_id)
    max_length = gen_length + input_ids.shape[1]
    
    # Handle block configuration
    if block_length is None:
        block_length = gen_length  # Default: single block (original behavior)
    
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"
 
    # Initialize cache for the prompt
    past_key_values = None

    # Process each block
    for num_block in range(num_blocks):
        
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # update cache
        if use_cache:
            model_output = model(x.repeat(draft_steps,1), attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            # Extract only previous block cache
            new_past_key_values = []
            for step in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[step])):
                    new_past_key_values[step] += (past_key_values[step][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values

        else:
            model_output = model(x.repeat(draft_steps,1), attention_mask, tok_idx, use_cache=False)
        
        logits = model_output.logits[:1, ...]
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, current_block_start] = x0[:, current_block_start]
        if histories is not None:
            histories.append(x.clone().cpu())

        if cgws is not None:
            window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)
            cur_x = x[:, window_slice].clone()
            cur_tok_idx = tok_idx[:, window_slice] if tok_idx is not None else None
        else:
            cur_x = x[:, current_block_start:].clone()
            cur_tok_idx = tok_idx[:, current_block_start:] if tok_idx is not None else None
        
        # Prepare attention mask for cached generation
        if attention_mask != "full":
            # Adjust attention mask for current position
            if cgws is not None:
                current_attention_mask = attention_mask[:, :, window_slice, :window_end]
            else:
                current_attention_mask = attention_mask[:, :, current_block_start:, :]
        else:
            current_attention_mask = attention_mask

        denoising_steps = steps_per_block[num_block]
        step = 1
        while True:
            
            
            mask_index = (cur_x == mask_token_id)
            mask_index[:, block_length:] = False

            x0, x0_p = sample_step(
                model, 
                cur_x.repeat(draft_steps,1), 
                current_block_start, current_attention_mask, cur_tok_idx, mask_index,
                past_key_values, use_cache,
                temperature, top_p, top_k, tar,
            )
            x0 = x0[:1, ...]
            x0_p = x0_p[:1, ...]
            
            if (cur_x[:, :block_length] == mask_token_id).sum() == 0:
                break

            cur_x = token_filling(
                    cur_x, x0, x0_p,
                    step, denoising_steps, 1, timesteps[num_block],
                    pad_token_id, pad_target_penalty,
                    mask_index, unmask_threshold,
                    block_length,
                    alg_temp,
                )

            if cgws is not None:
                x[:, window_slice] = cur_x
            else:
                x[:, current_block_start:] = cur_x
            
            if histories is not None:
                histories.append(x.clone().cpu())

            step += 1

            if (cur_x[:, :block_length] == mask_token_id).sum() == 0:
                break
        
        # block_all_pad = torch.all(
        #     x[:, current_block_start:current_block_end] == pad_token_id
        # )
        # if block_all_pad:
        #     if current_block_end < x.size(1):
        #         x[:, current_block_end:] = pad_token_id
        #     if histories is not None:
        #         histories.append(x.clone().cpu())
        #     break

    
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x


@torch.no_grad()
def block_diffusion_generate_FreeDave(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: Optional[float] = 0.9
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    gen_length = generation_config.max_gen_length
    steps = generation_config.steps
    draft_steps = generation_config.draft_steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, gen_length), value=mask_token_id)
    max_length = gen_length + input_ids.shape[1]
    
    # Handle block configuration
    if block_length is None:
        block_length = gen_length  # Default: single block (original behavior)
    
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]; broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"

    # # Initialize cache for the prompt
    past_key_values = None

    # Process each block
    for num_block in range(num_blocks):
        
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # update cache
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            # Extract only previous block cache
            new_past_key_values = []
            for step in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[step])):
                    new_past_key_values[step] += (past_key_values[step][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values

        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)

        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        x0_p, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, current_block_start] = x0[:, current_block_start]
        if histories is not None:
            histories.append(x.clone().cpu())

        if cgws is not None:
            window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)
            cur_x = x[:, window_slice].clone()
            cur_tok_idx = tok_idx[:, window_slice] if tok_idx is not None else None
        else:
            cur_x = x[:, current_block_start:].clone()
            cur_tok_idx = tok_idx[:, current_block_start:] if tok_idx is not None else None

        # Prepare attention mask for cached generation
        if attention_mask != "full":
            # Adjust attention mask for current position
            if cgws is not None:
                current_attention_mask = attention_mask[:, :, window_slice, :window_end]
            else:
                current_attention_mask = attention_mask[:, :, current_block_start:, :]
        else:
            current_attention_mask = attention_mask
        
        mask_index = (cur_x == mask_token_id)
        mask_index[:, block_length:] = False
        x0, x0_p = sample_step(
            model, 
            cur_x, 
            current_block_start, current_attention_mask, cur_tok_idx, mask_index,
            past_key_values, use_cache,
            temperature, top_p, top_k, tar,
        )

        denoising_steps = steps_per_block[num_block]
        step = 1
        while step < denoising_steps:

            cur_draft_steps = min(draft_steps, denoising_steps - step)
            # if cur_draft_steps == 1:
            # if denoising_steps - step == 1:
            #     cur_x = token_filling(
            #         cur_x, x0, x0_p,
            #         step, denoising_steps, cur_draft_steps, timesteps[num_block],
            #         pad_token_id, pad_target_penalty,
            #         mask_index, unmask_threshold,
            #         block_length,
            #         alg_temp,
            #     )
            #     break

            if (cur_x[:, :block_length] == mask_token_id).sum() == 0:
                break

            x_draft = token_filling(
                cur_x, x0, x0_p,
                step, denoising_steps, cur_draft_steps, timesteps[num_block],
                pad_token_id, pad_target_penalty,
                mask_index, unmask_threshold,
                block_length,
                alg_temp,
            )

            if denoising_steps - step > 1:
                if use_cache:
                    past_key_values = cache_batch_repeat_interleave(past_key_values, cur_draft_steps)

                mask_index_draft = (x_draft == mask_token_id)
                mask_index_draft[:, block_length:] = False
                x0_draft, x0_p_draft = sample_step(
                    model, 
                    x_draft, 
                    current_block_start, current_attention_mask, cur_tok_idx, mask_index_draft,
                    past_key_values, use_cache,
                    temperature, top_p, top_k, tar,
                )

                x_target = token_filling(
                    x_draft, x0_draft, x0_p_draft,
                    step+1, denoising_steps, 1, timesteps[num_block],
                    pad_token_id, pad_target_penalty,
                    mask_index_draft, unmask_threshold,
                    block_length,
                    alg_temp,
                )
                
                x_draft = x_draft.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])
                x_target = x_target.view(cur_x.shape[0], cur_draft_steps, *cur_x.shape[1:])
                matched_draft_index = (x_target[:, :-1, :] == x_draft[:, 1:, :]).all(dim=-1)
                matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1) #NOTE: assert matched_steps.shape[0] == 1 for now
                matched_steps.zero_() #NOTE: debug
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

            mask_index = (cur_x == mask_token_id)
            mask_index[:, block_length:] = False
            
            if (cur_x[:, :block_length] == mask_token_id).sum() == 0:
                break


        # # update cache
        # if use_cache:
        #     model_output = model(x, attention_mask, tok_idx, use_cache=True)
        #     past_key_values = model_output.past_key_values
        #     # Extract only previous block cache
        #     new_past_key_values = []
        #     for step in range(len(past_key_values)):
        #         new_past_key_values.append(())
        #         for j in range(len(past_key_values[step])):
        #             new_past_key_values[step] += (past_key_values[step][j][:, :current_block_start, :],)
        #     past_key_values = new_past_key_values

        block_all_pad = torch.all(
            x[:, current_block_start:current_block_end] == pad_token_id
        )
        # if block_all_pad:
        #     if current_block_end < x.size(1):
        #         x[:, current_block_end:] = pad_token_id
        #     if histories is not None:
        #         histories.append(x.clone().cpu())
        #     break

    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
        )
    else:
        return x

@torch.no_grad()
def sample_step(
    model,
    x,
    current_block_start, current_attention_mask, cur_tok_idx, mask_index,
    past_key_values, use_cache,
    temperature, top_p, top_k, tar,
):
    if use_cache:
        model_output = model(x, current_attention_mask, cur_tok_idx, 
                            past_key_values=past_key_values, use_cache=True)
        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
    else:
        model_output = model(x, current_attention_mask, cur_tok_idx, use_cache=False)
        logits = model_output.logits
        logits = logits[:, current_block_start:]
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

    # mask_logits = logits[mask_index]
    x0_p, x0 = sample_tokens(logits, temperature, top_p=top_p, top_k=top_k, tar=tar)

    return x0, x0_p

@torch.no_grad()
def token_filling(
    x, x0, x0_p,
    step, denoising_steps, draft_steps, timesteps,
    pad_token_id, pad_target_penalty,
    mask_index, unmask_threshold,
    block_length,
    alg_temp,
):

    x_draft = x.unsqueeze(1).expand(x.shape[0], draft_steps, *x.shape[1:]).clone()

    # —— pad token penalty ——
    _pad_target_divisor = pad_target_penalty
    _pad_mask_flat = (x0 == pad_token_id)  
    if _pad_mask_flat.any():
        x0_p = x0_p.clone()
        x0_p[_pad_mask_flat] = x0_p[_pad_mask_flat] / _pad_target_divisor

    confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=x0_p.dtype)
    confidence = confidence.float()
    confidence[mask_index] = x0_p[mask_index]
    confidence[:, block_length:] = -torch.inf
    # confidence = torch.where(mask_index, x0_p, -torch.inf)
    num_mask_token = mask_index.sum(dim=-1)

    if unmask_threshold is None:
        
        transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)
        for j in range(x_draft.shape[0]):
            for k in range(draft_steps):
                
                t = timesteps[step + k]
                s = timesteps[step + k + 1]
                number_transfer_tokens = int(num_mask_token[j] * (1 - s / t)) if step < denoising_steps - 1 else int(num_mask_token[j])
                
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, idx = torch.topk(confidence[j], number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        idx = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    
                    confidence[j, idx] = -torch.inf
                    transfer_index[j, k:, idx] = True
                    
    # TODO         
    # else:
    #     selected_map = torch.zeros_like(x, dtype=torch.bool)
    #     selected_map[mask_index] = (x0_p >= unmask_threshold)
    #     no_sel = ~selected_map.any(dim=-1)  # [B]
    #     no_sel = no_sel & mask_index.any(dim=-1)

    #     if no_sel.any():
    #         masked_scores = confidence.masked_fill(~mask_index, float("-inf"))
    #         best_idx = torch.argmax(masked_scores, dim=-1)
    #         selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
    #         selected_map[selected_rows, best_idx[selected_rows]] = True

    #     selected_map &= mask_index
    #     x_candidates = torch.full_like(x, mask_token_id, dtype=torch.long)
    #     x_candidates[mask_index] = x0

    #     x[selected_map] = x_candidates[selected_map]
    
    x_draft[transfer_index] = x0.unsqueeze(1).expand_as(x_draft)[transfer_index]
    x_draft = x_draft.view(x.shape[0] * draft_steps, *x.shape[1:]) # (batch * draft_steps, seq_len)
    return x_draft

@torch.no_grad()
def cache_batch_repeat_interleave(past_key_values, n):
    '''
        past_key_values: 
        a list of n_heads tuples (key, value)
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
            new_past_key_values[i] += (past_key_values[i][j].repeat(n, 1, 1),)
    return new_past_key_values

@torch.no_grad()
def cache_batch_select_indices(past_key_values, indices):
    '''
        past_key_values: 
        a list of n_heads tuples (key, value)
        key: (batch, seq_len, head_dim)
        value: (batch, seq_len, head_dim)
    '''
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            new_past_key_values[i] += (past_key_values[i][j][indices, ...],)
    return new_past_key_values