from __future__ import annotations
import math, json, os, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp


from sample.dream import DreamTokenizer
from sample.dream.modeling_dream import DreamModel
from sample.dream.generation_utils_block import DreamGenerationMixin
import types
from sample.dream.generation_utils_block import DreamGenerationConfig


from transformers.utils import ModelOutput
from typing import Any, Dict, Optional, Tuple, Union
import torch.distributions as dists
from dataclasses import dataclass
from torch.nn import functional as F
import torch


from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

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
    max_length = generation_config.max_gen_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    gen_length = max_length - input_ids.shape[1]
    
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
    max_length = generation_config.max_gen_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    gen_length = max_length - input_ids.shape[1]
    
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
                cur_x, 
                current_block_start, current_attention_mask, cur_tok_idx, mask_index,
                past_key_values, use_cache,
                temperature, top_p, top_k, tar,
            )
            
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

import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list


# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem = data_i["question"])



def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"




def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output



def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]       
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0):  
        cur = snap[sample_idx]       
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:    
            break
    return step_map



from tqdm import tqdm




def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, batch_size, config):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load model once
    model_gpu = (DreamModel.from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())
    model_gpu.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model_gpu)
    model_gpu._sample = types.MethodType(DreamGenerationMixin._sample, model_gpu)   
    tokenizer_gpu = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    pad_id = model_gpu.config.pad_token_id
    mask_id = model_gpu.config.mask_token_id
    eos_id = tokenizer_gpu.convert_tokens_to_ids("<|im_end|>")

    # process in chunks of `batch_size`
    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        # tokenize & move to GPU
        enc = tokenizer_gpu(batch_prompts,
                            padding=True, #truncation=True,
                            return_tensors="pt", padding_side="left")
        prompt_ids = enc["input_ids"].to(device)

        attn_mask = prompt_ids.ne(pad_id)
        #attn_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        attn_mask = attn_mask.to(device=model_gpu.device)

        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None

        generation_config = DreamGenerationConfig(
            output_history=True,            
            return_dict_in_generate=True,   
            max_length=config.rollout.max_gen_length + prompt_ids.shape[1],     
            steps=config.rollout.steps,                  
            temperature=config.rollout.temperature,  
            top_p=config.rollout.top_p,               
            top_k=config.rollout.top_k,            
            tar=config.rollout.target,               
            alg_temp=config.rollout.alg_temp,        
        )
        
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        generation_ids = block_diffusion_generate(
            model_gpu,
            prompt_ids,
            attention_mask=attn_mask,
            generation_config=generation_config,
            block_length=config.rollout.block_size,
            use_cache=config.rollout.use_cache,
            further_horizon=config.rollout.further_horizon,
            mask_token_id = mask_id,
            eos_token_id = eos_id,
            pad_token_id = pad_id,
            pad_target_penalty = config.rollout.pad_target_penalty,
            unmask_threshold = unmask_threshold
        )
        generation_ids.sequences = generation_ids.sequences.cpu()
        torch.cuda.empty_cache()

        # decode
        seq_ids = generation_ids.sequences[:, prompt_ids.shape[1]:].tolist()
        texts   = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        # compute and store step maps
        for i, idx in enumerate(batch_idxs):
            # extract step map for sample i in this batch
            m = denoise_step_map(generation_ids.history, mask_id=mask_id, sample_idx=i)
            step_map = m[prompt_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map

        # free unused GPU cache
        torch.cuda.empty_cache()

def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


if __name__ == "__main__":

    config = get_config()

    mp.set_start_method("spawn", force=True)

    
    k_sample = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size
    system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
    
    project_name = config.experiment.project

    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
        system_prompts_stdio = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
    elif config.dataset.data_type == "option":
        system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant\n'''
    
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)
    #data = [data[i] for i in range(32)]

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)
    
    num = len(data)

    tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)




    # initialization
    generation_prompts = []
    prefix_list = []
    index_list = []
    for step in range(num):
        # preprocess
        if code_eval:
            if data[step]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[step]["prefix"]
                prefix_list = prefix_list + [data[step]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[step])] * k_sample
        
        index_list = index_list + [step] * k_sample
        data[step]["full_output"] = []
        data[step]["step_map"] = []
        data[step]["extracted_output"] = []
        data[step]["response_length"] = []
        data[step]["prompt"] = get_prompt(data[step])

    

    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    # --------------------- 2. split to each GPU ----------------------
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 1, "need >=2 GPUs for parallel inference"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks    = split_even(shuffled_idx,     n_gpu)

    

    # ------------------- 4. launch all workers -----------------------
    manager    = mp.Manager()
    seq_dict   = manager.dict()   # {shuffled_pos: text}
    step_dict  = manager.dict()   # {shuffled_pos: step_map}
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                    args=(pretrained_model, rk,
                            prompt_chunks[rk],
                            idx_chunks[rk],
                            seq_dict,
                            step_dict,
                            batch_size,
                            config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ------------------- 5. restore original order -------------------
    restored_outputs    = [seq_dict[i]  for i in range(N)]
    restored_step_maps  = [step_dict[i] for i in range(N)]

    cprint("generation job done!", "green")






    import re
    

    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token

        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped

        collapse_re = re.compile(pattern)

        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)




    # process generated codes
    step = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(step/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[step] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[step]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[step])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[step])
        step += 1

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


