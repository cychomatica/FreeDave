import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import Cache
from transformers.utils import logging
from typing import Optional, List, Union
from deprecated import deprecated
from .sampling_utils import sample_tokens
import math
from transformers.cache_utils import DynamicCache
from .cache_utils import DynamicDualCache
from .determinism_utils import deterministic_sdpa
from dataclasses import dataclass
from contextlib import nullcontext

logger = logging.get_logger(__name__)

# NOTE: now supporting Dream, SDAR, TraDo, and LLaDA (with sdpa_additive_attention_mask where needed).

# TODO: support more AR-adapted DLM
AR_ADAPTED_DLM = ['DreamModel', 'DreamBaseModel']


def _mask_is_binary_visibility(am: torch.Tensor) -> bool:
    if am.dim() != 4:
        return False
    if am.dtype == torch.bool:
        return True
    if not am.is_floating_point():
        return False
    return bool(torch.logical_or(am == 0, am == 1).all().item())


def visibility_mask_to_sdpa_additive(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert a 4D *visibility* mask (1/0 float or bool; 1 = attend) into an SDPA *additive* bias
    (0 = keep logit, finfo.min = block). Idempotent for masks that are already additive (not pure {0,1}).
    """
    if attention_mask.dim() != 4 or not _mask_is_binary_visibility(attention_mask):
        return attention_mask
    finfo_min = torch.finfo(dtype).min
    if attention_mask.dtype == torch.bool:
        return torch.where(
            attention_mask,
            torch.zeros((), dtype=dtype, device=attention_mask.device),
            torch.full((), finfo_min, dtype=dtype, device=attention_mask.device),
        )
    am = attention_mask
    return torch.where(
        am > 0.5,
        torch.zeros((), dtype=dtype, device=am.device),
        torch.full((), finfo_min, dtype=dtype, device=am.device),
    ).expand_as(am)


def is_ar_adapted_dlm_model(model: nn.Module) -> bool:
    """
    True if the backbone is Dream-style and needs the AR-adapted logits shift in
    `get_processed_model_outputs`. PEFT, DeepSpeed, and DDP wrap the real module;
    a plain `model.__class__.__name__` check misses those and breaks generation.
    """
    seen: set[int] = set()
    m: Optional[nn.Module] = model
    for _ in range(16):
        if m is None or id(m) in seen:
            break
        seen.add(id(m))
        if m.__class__.__name__ in AR_ADAPTED_DLM:
            return True
        nxt: Optional[nn.Module] = None
        if hasattr(m, "get_base_model"):
            try:
                nxt = m.get_base_model()
            except Exception:
                nxt = None
        if nxt is not None and nxt is not m:
            m = nxt
            continue
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            inner = m.base_model.model
            if inner is not m:
                m = inner
                continue
        if hasattr(m, "module"):
            inner = m.module
            if inner is not m:
                m = inner
                continue
        break
    return False

@dataclass
class DLMGenerationOutput:
    sequences: torch.Tensor = None
    trajectory_step_map: torch.Tensor = None
    hit_rates: Optional[torch.Tensor] = None


class DLMGeneration:

    def __init__(
        self,
        sdpa_additive_attention_mask: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        sdpa_backend: Optional[str] = None,
    ):
        """
        Args:
            sdpa_additive_attention_mask: If True, 4D attention masks are converted from visibility (1/0 or bool)
                to SDPA additive form before each model forward. Use for backbones that do not convert masks
                internally (e.g. LLaDA/Dream SDPA). Keep False for TraDo/SDAR, which expect 1/0 and convert
                inside `SDARAttention`.
            seed: Random seed for reproducibility (unused here; kept for caller convenience).
            deterministic: If True, force a single SDPA backend for every forward pass so that baseline and
                FreeDave follow identical floating-point code-paths (eliminates Flash-vs-Math kernel divergence).
            sdpa_backend: SDPA backend to force when ``deterministic=True``.  One of ``"math"`` (default/safest),
                ``"mem_efficient"``, or ``"flash"``.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_per_step = 1
        self.sdpa_additive_attention_mask = sdpa_additive_attention_mask
        self.seed = seed
        self.deterministic = deterministic
        self.sdpa_backend = sdpa_backend

    def _sdpa_ctx(self):
        """Return a context manager that pins the SDPA backend when deterministic mode is on."""
        if self.deterministic:
            return deterministic_sdpa(self.sdpa_backend)
        return nullcontext()

    def get_processed_model_outputs(
        self, 
        model: nn.Module,
        x: torch.Tensor, 
        attention_mask: Optional[torch.LongTensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: bool = False,
        store_kv: bool = False,
        output_attentions: bool = False,
    ):

        '''
            For AR-adapted DLM, we need to process the model logits due to the logits shift.
        '''

        if self.sdpa_additive_attention_mask and attention_mask is not None:
            try:
                _w_dtype = next(model.parameters()).dtype
            except StopIteration:
                _w_dtype = torch.float32
            attention_mask = visibility_mask_to_sdpa_additive(attention_mask, _w_dtype)

        with self._sdpa_ctx():
            if is_ar_adapted_dlm_model(model):
                model_output = model(
                    input_ids=x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                attentions = model_output.attentions if output_attentions and hasattr(model_output, 'attentions') else None

                return logits, attentions

            else:
                model_output = model(
                    input_ids=x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    store_kv=store_kv,
                )
                logits = model_output.logits
                attentions = model_output.attentions if output_attentions and hasattr(model_output, 'attentions') else None

                return logits, attentions

    def get_attention_mask_and_position_ids(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        block_length: Optional[int] = None, 
        num_blocks: Optional[int] = None, 
        attention_type: Optional[str] = None
    ):
        
        if attention_type == 'full':
            if attention_mask is not None and torch.any(attention_mask == 0.0): # when input_ids is padded
                attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                attention_mask = torch.logical_and(
                    attention_mask.unsqueeze(1).unsqueeze(-2),
                    attention_mask.unsqueeze(1).unsqueeze(-1),
                )
            else:
                attention_mask = None
                position_ids = None
        elif attention_type == 'block_causal':
            if block_length is None or num_blocks is None:
                raise ValueError('block_length and num_blocks must be provided when attention_type is block_causal')
            inter_block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
            attention_mask = inter_block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0).unsqueeze(0) # (1, 1, num_blocks * block_length, num_blocks * block_length)
            position_ids = torch.arange(max_length, device=self.device).unsqueeze(0)
        else:
            logger.warning(f'Attention type {attention_type} not supported, using full attention (attention_mask = None) by default.')
            attention_mask = None
            position_ids = None
        return attention_mask, position_ids

    def create_tree_attention_mask_and_position_ids(
        self, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        shared_prefix_length: Optional[int] = None,
        branch_length: Optional[int] = None,
        shared_suffix_length: Optional[int] = 0,
        num_branches: Optional[int] = None
    ):
        '''
        Create tree-structured attention mask and position ids for draft-tree verification.
        Supports both block-causal and full-attention settings.
        '''
        if (
            shared_prefix_length is None
            or branch_length is None
            or shared_suffix_length is None
            or num_branches is None
        ):
            raise ValueError("Tree attention requires shared_prefix_length/branch_length/shared_suffix_length/num_branches.")

        expected_width = shared_prefix_length + branch_length + shared_suffix_length

        if attention_mask is None:
            attn_mask = torch.ones(
                branch_length,
                expected_width,
                device=self.device,
                dtype=torch.bool,
            )
        else:
            if attention_mask.ndim == 4:
                attn_mask = attention_mask.squeeze(0).squeeze(0)
            elif attention_mask.ndim == 2:
                attn_mask = attention_mask
            else:
                raise ValueError("attention_mask must be 2D or 4D when creating tree attention.")
            if attn_mask.shape != (branch_length, expected_width):
                raise ValueError(
                    f"attention_mask shape mismatch: got {tuple(attn_mask.shape)}, "
                    f"expected ({branch_length}, {expected_width})."
                )

        if position_ids is None:
            position_ids = torch.arange(shared_prefix_length, shared_prefix_length + branch_length, device=self.device).unsqueeze(0)

        tree_attention_mask = torch.zeros(
            branch_length * num_branches,
            shared_prefix_length + branch_length * num_branches + shared_suffix_length,
            device=attn_mask.device,
            dtype=attn_mask.dtype,
        )

        for branch_idx in range(num_branches):
            row_start = branch_idx * branch_length
            row_end = row_start + branch_length

            if shared_prefix_length > 0:
                tree_attention_mask[row_start:row_end, :shared_prefix_length] = attn_mask[:, :shared_prefix_length]

            col_start = shared_prefix_length + branch_idx * branch_length
            col_end = col_start + branch_length
            tree_attention_mask[row_start:row_end, col_start:col_end] = attn_mask[
                :, shared_prefix_length:shared_prefix_length + branch_length
            ]

            if shared_suffix_length > 0:
                tree_attention_mask[row_start:row_end, -shared_suffix_length:] = attn_mask[
                    :, shared_prefix_length + branch_length:
                ]

        tree_position_ids = position_ids.repeat(1, num_branches)
        tree_attention_mask = tree_attention_mask.unsqueeze(0).unsqueeze(0)
        return tree_attention_mask, tree_position_ids


    @deprecated('This function is deprecated, now only consider 1 token per step')
    def get_num_transfer_tokens(block_length, steps):
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    def get_mask_token_prediction_and_confidence(
        self,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        mask_token_id: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):

        confidence = torch.full_like(mask_index, -torch.inf, device=self.device, dtype=logits.dtype)
        x0 = torch.full_like(mask_index, mask_token_id, device=self.device, dtype=torch.long)
        mask_logits = logits[mask_index]
        mask_x0, mask_confidence = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        confidence[mask_index] = mask_confidence
        x0[mask_index] = mask_x0
        return x0, confidence

    def token_transfer(
        self, 
        x: torch.Tensor, 
        x0: torch.Tensor, 
        confidence: torch.Tensor, 
        block_length: int,
        mask_token_id: int, 
        confidence_priority_shift: Optional[torch.Tensor] = None,
        alg_temp: Optional[float] = None,
        confidence_threshold: Optional[float] = None, 
        token_per_step: Optional[int] = 1,
        draft_steps: Optional[int] = None,
    ):
        '''
        Returns the updated x after token transfer.
        Args:
            x: torch.Tensor, the original sequence
            x0: torch.Tensor, the sampled token
            confidence: torch.Tensor, the confidence scores of the sampled tokens
            mask_token_id: int, the mask token id
            alg_temp: Optional[float], the temperature for token transferring
            confidence_threshold: Optional[float], the confidence threshold for token transferring
            num_transfer_tokens: Union[torch.Tensor, List[int]], a 1-D tensor representing the numbers of transfer tokens. If len(num_transfer_tokens) > 1, multiple draft sequences will be decoded.
        '''

        mask_index = (x == mask_token_id)

        if confidence_threshold is not None:
            high_confidence_index = (confidence >= confidence_threshold)
            
            if draft_steps is None or draft_steps == 1:
                # confidence-aware parallel decoding
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                
                if alg_temp is None or alg_temp == 0:
                    for j in range(confidence.shape[0]):
                        if high_confidence_index[j].any():
                            transfer_index[j, high_confidence_index[j]] = True
                        else:
                            _, idx = torch.topk(confidence[j], min(token_per_step, mask_index[j].sum()))
                            transfer_index[j, idx] = True
                    x[transfer_index] = x0[transfer_index]
                    return x
                elif alg_temp == -1: # find the left-to-right consecutive tokens with highest confidence; if none, unmask the leftmost token
                    for j in range(confidence.shape[0]):
                        mask_high_confidence_index = high_confidence_index[j, mask_index[j]]
                        if mask_high_confidence_index.cumprod(dim=-1).sum().item() > 0:
                            idx = torch.arange(x.size(1), device=self.device)[mask_index[j]][:mask_high_confidence_index.cumprod(dim=-1).sum().item()]
                        else:
                            idx = torch.arange(x.size(1), device=self.device)[mask_index[j]][:token_per_step]
                        transfer_index[j, idx] = True
                    x[transfer_index] = x0[transfer_index]
                    return x
                else:
                    raise NotImplementedError(f'Algorithm temperature {alg_temp} not supported yet.')
            else:
                # TODO: dynamic draft_steps based on confidence threshold, drafting all tokens with confidence >= confidence_threshold
                # if num_high_confidence < draft_steps, still use draft_steps, otherwise use num_high_confidence
                # TODO: additional draft on high confidence tokens
                raise NotImplementedError('Not implemented yet.')

        else:
            if draft_steps is not None and draft_steps > 1:
                x_draft = x.unsqueeze(1).expand(x.shape[0], draft_steps, *x.shape[1:]).clone() # (batch, draft_steps, seq_len)
            else:
                draft_steps = 1
                x_draft = x.unsqueeze(1).clone() # (batch, 1, seq_len)
            transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)

            if confidence_priority_shift is not None:
                confidence = confidence + confidence_priority_shift

            if alg_temp is None or alg_temp == 0: # maskgit: unmask tokens with highest confidence
                for j in range(confidence.shape[0]):
                    for k in range(draft_steps):
                        _, idx = torch.topk(confidence[j], min(token_per_step * (k+1), mask_index[j].sum()))
                        transfer_index[j, k, idx] = True
            elif alg_temp == -1: # l2r: unmask left to right
                for j in range(confidence.shape[0]):
                    for k in range(draft_steps):
                        idx = torch.arange(x.size(1), device=self.device)[mask_index[j]][:token_per_step * (k+1)]
                        transfer_index[j, k, idx] = True
            else:
                raise NotImplementedError(f'Algorithm temperature {alg_temp} not supported yet. ')

            if draft_steps is not None and draft_steps > 1:
                x_draft[transfer_index] = x0.unsqueeze(1).expand_as(x_draft)[transfer_index] # (batch, draft_steps, seq_len)
                x_draft = x_draft.view(x.shape[0] * draft_steps, *x.shape[1:]) # (batch * draft_steps, seq_len)
            else:
                x_draft[transfer_index] = x0.unsqueeze(1)[transfer_index] # (batch, 1, seq_len)
                x_draft = x_draft.squeeze(1) # (batch, seq_len)

            return x_draft

    @torch.no_grad()
    def block_decode_with_full_attention(
        self, 
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        use_cache: bool = False,
        dual_cache: bool = False,
        mask_token_id: int = 151666,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
        early_exit: Optional[bool] = False
    ):
        '''
        NOTE: now only consider 1 token per step; all the token transfer operations are performed in the block region.

        Returns:
            x: ``[batch, prompt_length + max_gen_length]`` decoded ids.
            generated_trajectory_step_map: ``[batch, max_gen_length]`` long tensor, step index at which each
            completion position was first unmasked (``-1`` if still masked when the decode budget ends).
        '''

        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]
        max_length = prompt_length + max_gen_length
        x = F.pad(input_ids, (0, max_gen_length), value=mask_token_id)
        trajectory_step_map = torch.full((batch_size, max_length), -1, device=x.device, dtype=torch.long)
        global_step = 0

        if block_length is None:
            block_length = max_gen_length

        assert max_gen_length % block_length == 0, 'max_gen_length {} must be divisible by block_length {}'.format(max_gen_length, block_length)
        num_gen_blocks = max_gen_length // block_length

        assert decoding_steps % num_gen_blocks == 0, 'decoding_steps {} must be divisible by num_gen_blocks {}'.format(decoding_steps, num_gen_blocks)
        steps_per_block = decoding_steps // num_gen_blocks
        # assert steps_per_block == block_length, 'steps_per_block must be equal to block_length for now'
        # num_transfer_tokens_per_block = self.get_num_transfer_tokens(block_length, steps_per_block)

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_gen_blocks,
            attention_type='full'
        )
        # past_key_values = None

        for block_idx in range(num_gen_blocks):

            current_block_start = prompt_length + block_idx * block_length
            current_block_end = current_block_start + block_length
            current_block_mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)

            # update cache and decode the first step in the current block
            # reset cache to empty
            if dual_cache:
                past_key_values = DynamicDualCache()
            else:
                past_key_values = DynamicCache()
            # update cache and get logits
            logits, _ = self.get_processed_model_outputs(
                model=model,
                x=x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=False
            )

            if is_ar_adapted_dlm_model(model):
                # always unmask the first token in the current block due to the logits shift of AR-adapted DLM
                x0, confidence = sample_tokens(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                x[:, current_block_start] = x0[:, current_block_start]
                trajectory_step_map[:, current_block_start] = global_step
            else:
                current_block_x0, current_block_confidence = self.get_mask_token_prediction_and_confidence(
                    logits=logits[:, current_block_start:current_block_end, :],
                    mask_index=current_block_mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                x[:, current_block_start:current_block_end] = self.token_transfer(
                    x=x[:, current_block_start:current_block_end],
                    x0=current_block_x0,
                    confidence=current_block_confidence,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                    confidence_threshold=confidence_threshold
                )
                newly_unmasked = current_block_mask_index & (
                    x[:, current_block_start:current_block_end] != mask_token_id
                )
                trajectory_step_map[:, current_block_start:current_block_end][newly_unmasked] = global_step

            global_step += 1

            if dual_cache:
                # Compose [prefix, current-block, suffix] on-the-fly from base cache.
                past_key_values.set_dual_layout(
                    prefix_end=current_block_start,
                    suffix_start=current_block_end,
                )
            else: # Extract only previous block cache
                past_key_values.crop(current_block_start)

            # decoding loop for the current block
            step = 1
            while step < steps_per_block:

                current_block_mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                
                # Prepare attention mask for cached generation
                if attention_mask != "full" and attention_mask is not None:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
                else:
                    current_attention_mask = attention_mask
                
                if dual_cache:
                    # use the current block for forward pass
                    logits, _ = self.get_processed_model_outputs(
                        model=model,
                        x=x[:, current_block_start:current_block_end], 
                        attention_mask=current_attention_mask, 
                        position_ids=torch.arange(current_block_start, current_block_end, device=x.device).unsqueeze(0), 
                        past_key_values=past_key_values, 
                        use_cache=use_cache
                    )
                else:
                    # use the entire remaining sequence for forward pass
                    logits, _ = self.get_processed_model_outputs(
                        model=model,
                        x=x[:, current_block_start:], 
                        attention_mask=current_attention_mask, 
                        position_ids=position_ids[:, current_block_start:] if position_ids is not None else None, 
                        past_key_values=past_key_values, 
                        use_cache=use_cache
                    )
                    past_key_values.crop(current_block_start)

                current_block_x0, current_block_confidence = self.get_mask_token_prediction_and_confidence(
                    logits=logits[:, :block_length, :],
                    mask_index=current_block_mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

                x[:, current_block_start:current_block_end] = self.token_transfer(
                    x=x[:, current_block_start:current_block_end],
                    x0=current_block_x0,
                    confidence=current_block_confidence,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                    confidence_threshold=confidence_threshold
                )
                newly_unmasked = current_block_mask_index & (
                    x[:, current_block_start:current_block_end] != mask_token_id
                )
                if newly_unmasked.any():
                    trajectory_step_map[:, current_block_start:current_block_end][newly_unmasked] = global_step
                    global_step += 1

                step += 1

                if (x[:, current_block_start:current_block_end] != mask_token_id).all():
                    break

            if early_exit and (x[:, current_block_start:current_block_end] == eos_token_id).all():
                if current_block_end < max_length:
                    x[:, current_block_end:] = eos_token_id
                    assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
                break

            if early_exit and (x[:, current_block_start:current_block_end] == pad_token_id).all():
                if current_block_end < max_length:
                    x[:, current_block_end:] = pad_token_id
                    assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
                break

        generated_trajectory_step_map = trajectory_step_map[:, prompt_length:]
        # assert (generated_trajectory_step_map >= 0).all(), 'Undecoded positions found: {}.'.format((generated_trajectory_step_map < 0).nonzero())

        return DLMGenerationOutput(sequences=x, trajectory_step_map=generated_trajectory_step_map)

    @torch.no_grad()
    def block_decode_with_full_attention_FreeDave(
        self, 
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        draft_steps: Optional[int] = 1,
        eager_acceptance_mode: Optional[bool] = False,
        draft_mode: Optional[str] = 'tree_attention',
        use_cache: bool = False,
        dual_cache: bool = False,
        mask_token_id: int = 151666,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
        early_exit: Optional[bool] = False,
    ):
        '''
        block_decode_with_full_attention_FreeDave for full-attention DLMs.
        Uses draft-and-verify with tree attention, while supporting:
        1) prefix cache (DynamicCache) and
        2) dual cache (DynamicDualCache).
        '''

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'batch size must be 1 for FreeDave'

        prompt_length = input_ids.shape[1]
        max_length = prompt_length + max_gen_length
        x = F.pad(input_ids, (0, max_gen_length), value=mask_token_id)
        trajectory_step_map = torch.full((batch_size, max_length), -1, device=x.device, dtype=torch.long)

        if block_length is None:
            block_length = max_gen_length

        assert max_gen_length % block_length == 0, 'max_gen_length {} must be divisible by block_length {}'.format(max_gen_length, block_length)
        num_gen_blocks = max_gen_length // block_length

        assert block_length % self.token_per_step == 0, 'block_length={} must be divisible by self.token_per_step={}'.format(block_length, self.token_per_step)
        steps_per_block = block_length // self.token_per_step

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_gen_blocks,
            attention_type='full'
        )
        global_step = 0

        for block_idx in range(num_gen_blocks):

            current_block_start = prompt_length + block_idx * block_length
            current_block_end = current_block_start + block_length

            if eager_acceptance_mode:
                num_future_blocks = min(math.ceil(draft_steps / steps_per_block), num_gen_blocks - block_idx - 1)
                confidence_priority_shift = torch.arange(num_future_blocks, -1, -1, device=x.device).repeat_interleave(block_length)
            else:
                num_future_blocks = 0
                confidence_priority_shift = None

            current_window_start = current_block_start
            current_window_end = current_window_start + (num_future_blocks + 1) * block_length
            current_window_len = current_window_end - current_window_start

            # current_window_x = x[:, current_window_start:current_window_end].clone()
            # current_window_mask_index = (current_window_x == mask_token_id)

            # if current_window_mask_index[:, :block_length].sum() == 0:
            #     continue

            if dual_cache:
                past_key_values = DynamicDualCache()
            else:
                past_key_values = DynamicCache()
            # update cache and get logits
            logits, _ = self.get_processed_model_outputs(
                model=model,
                x=x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=False
            )
            if is_ar_adapted_dlm_model(model):
                # always unmask the first token in the current block due to the logits shift of AR-adapted DLM
                x0, confidence = sample_tokens(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                x[:, current_block_start] = x0[:, current_block_start]
                trajectory_step_map[:, current_block_start] = global_step

            if (x[:, current_block_start:current_block_end] != mask_token_id).all():
                continue
            
            if dual_cache:
                # Compose [prefix, current-window, suffix] from the frozen base cache.
                past_key_values.set_dual_layout(
                    prefix_end=current_window_start,
                    suffix_start=current_window_end,
                )
            else: # Extract only previous block cache
                past_key_values.crop(current_block_start)

            current_window_mask_index = (x[:, current_window_start:current_window_end] == mask_token_id)

            # reuse stale logits from the cache-update forward
            current_window_x0, current_window_confidence = self.get_mask_token_prediction_and_confidence(
                logits=logits[:, current_window_start:current_window_end, :],
                mask_index=current_window_mask_index,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            step = 1

            while True:

                current_window_mask_index = (x[:, current_window_start:current_window_end] == mask_token_id)
                current_window_draft_steps = min(draft_steps, math.ceil(current_window_mask_index.sum().item() / self.token_per_step))

                if step == 1 and is_ar_adapted_dlm_model(model):
                    # one fewer draft if reuse the cache update logits
                    if current_window_draft_steps > 1:
                        current_window_x_draft = self.token_transfer(
                            x=x[:, current_window_start:current_window_end],
                            x0=current_window_x0,
                            draft_steps=current_window_draft_steps-1,
                            confidence=current_window_confidence,
                            confidence_priority_shift=confidence_priority_shift,
                            block_length=block_length,
                            mask_token_id=mask_token_id,
                            alg_temp=alg_temp,
                            token_per_step=self.token_per_step,
                        )
                        # prepend the known-good current x as the first draft
                        current_window_x_draft = torch.cat([x[:, current_window_start:current_window_end].clone(), current_window_x_draft], dim=0)
                    else:
                        current_window_x_draft = x[:, current_window_start:current_window_end].clone()
                    
                else:
                    current_window_x_draft = self.token_transfer(
                        x=x[:, current_window_start:current_window_end],
                        x0=current_window_x0,
                        draft_steps=current_window_draft_steps,
                        confidence=current_window_confidence,
                        confidence_priority_shift=confidence_priority_shift,
                        block_length=block_length,
                        mask_token_id=mask_token_id,
                        alg_temp=alg_temp,
                        token_per_step=self.token_per_step,
                    )
                assert current_window_x_draft.shape == (batch_size * current_window_draft_steps, current_window_len), 'current_window_x_draft.shape must be (batch_size * current_window_draft_steps, current_window_len), but got {}'.format(current_window_x_draft.shape)

                if num_gen_blocks - block_idx == 1 and current_window_mask_index[:, :block_length].sum() <= self.token_per_step:
                    x[:, current_window_start:current_window_end] = current_window_x_draft[::current_window_draft_steps]
                    newly_unmasked = current_window_mask_index & (x[:, current_window_start:current_window_end] != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                    break

                if not eager_acceptance_mode and current_window_mask_index[:, :block_length].sum() <= self.token_per_step:
                    x[:, current_window_start:current_window_end] = current_window_x_draft[::current_window_draft_steps]
                    newly_unmasked = current_window_mask_index & (x[:, current_window_start:current_window_end] != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                    break
                
                if dual_cache:

                    if draft_mode == 'batch_expanding':
                        past_key_values.batch_repeat_interleave(current_window_draft_steps)
                        logits, _ = self.get_processed_model_outputs(
                            model=model,
                            x=current_window_x_draft,
                            attention_mask=None,
                            position_ids=torch.arange(current_window_start, current_window_end, device=x.device).unsqueeze(0).expand(batch_size * current_window_draft_steps, -1),
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=False
                        )
                        past_key_values.batch_select_indices(torch.arange(0, current_window_x_draft.shape[0], current_window_draft_steps))

                        current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits,
                            mask_index=(current_window_x_draft == mask_token_id),
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k
                        )
                    elif draft_mode == 'tree_attention':
                        tree_attention_mask, tree_position_ids = self.create_tree_attention_mask_and_position_ids(
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            shared_prefix_length=current_window_start,
                            branch_length=current_window_len,
                            shared_suffix_length=max_length-current_window_end,
                            num_branches=current_window_draft_steps
                        )
                        logits, _ = self.get_processed_model_outputs(
                            model=model,
                            x=current_window_x_draft.view(batch_size, -1),
                            attention_mask=tree_attention_mask,
                            position_ids=tree_position_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=False
                        )
                        current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits.view(current_window_x_draft.shape[0], -1, logits.shape[-1]),
                            mask_index=(current_window_x_draft == mask_token_id),
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k
                        )
                    else:
                        raise ValueError(f'Invalid draft mode: {draft_mode}. Please choose from "batch_expanding" or "tree_attention".')
                else:
                    if draft_mode == 'batch_expanding':
                        past_key_values.batch_repeat_interleave(current_window_draft_steps)
                        logits, _ = self.get_processed_model_outputs(
                            model=model,
                            x=torch.cat([current_window_x_draft, x[:, current_window_end:].expand(current_window_x_draft.shape[0], -1)], dim=-1),
                            attention_mask=attention_mask,
                            position_ids=position_ids[:, current_block_start:] if position_ids is not None else None,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=False
                        )
                        past_key_values.batch_select_indices(torch.arange(0, current_window_x_draft.shape[0], current_window_draft_steps))
                        past_key_values.crop(current_block_start)

                        current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits[:, :current_window_len, :],
                            mask_index=(current_window_x_draft == mask_token_id),
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k
                        )

                    elif draft_mode == 'tree_attention':
                        tree_attention_mask, tree_position_ids = self.create_tree_attention_mask_and_position_ids(
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            shared_prefix_length=current_window_start,
                            branch_length=max_length-current_window_start,
                            shared_suffix_length=0,
                            num_branches=current_window_draft_steps
                        )

                        logits, _ = self.get_processed_model_outputs(
                            model=model,
                            x=torch.cat([current_window_x_draft, x[:, current_window_end:].expand(current_window_x_draft.shape[0], -1)], dim=-1).view(batch_size, -1),
                            attention_mask=tree_attention_mask,
                            position_ids=tree_position_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=False
                        )
                        past_key_values.crop(current_block_start)

                        current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits.view(current_window_x_draft.shape[0], -1, logits.shape[-1])[:, :current_window_len, :],
                            mask_index=(current_window_x_draft == mask_token_id),
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k
                        )

                step += 1

                current_window_x_target = self.token_transfer(
                    x=current_window_x_draft,
                    x0=current_window_x0_draft,
                    confidence=current_window_confidence_draft,
                    confidence_priority_shift=confidence_priority_shift,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                )

                current_window_x_draft = current_window_x_draft.view(
                    batch_size, current_window_draft_steps, current_window_len
                )
                current_window_x_target = current_window_x_target.view(
                    batch_size, current_window_draft_steps, current_window_len
                )
                matched_draft_index = (current_window_x_target[:, :-1, :] == current_window_x_draft[:, 1:, :]).all(dim=-1)
                matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1).item()

                if eager_acceptance_mode and (current_window_x_draft[:, matched_steps, :block_length] != mask_token_id).all():
                    x[:, current_window_start:current_window_end] = current_window_x_target[:, matched_steps, :]
                    newly_unmasked = current_window_mask_index & (x[:, current_window_start:current_window_end] != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                else:
                    x[:, current_window_start:current_window_end] = current_window_x_draft[:, matched_steps, :]
                    newly_unmasked = current_window_mask_index & (x[:, current_window_start:current_window_end] != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                    current_window_x0 = current_window_x0_draft.view(
                        batch_size, current_window_draft_steps, current_window_len
                    )[:, matched_steps, :]
                    current_window_confidence = current_window_confidence_draft.view(
                        batch_size, current_window_draft_steps, current_window_len
                    )[:, matched_steps, :]

                if (x[:, current_window_start:current_window_end] != mask_token_id).all():
                    break

            if early_exit and (x[:, current_block_start:current_block_end] == eos_token_id).all():
                if current_block_end < max_length:
                    x[:, current_block_end:] = eos_token_id
                    assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
                break

            if early_exit and (x[:, current_block_start:current_block_end] == pad_token_id).all():
                if current_block_end < max_length:
                    x[:, current_block_end:] = pad_token_id
                    assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
                break

        generated_trajectory_step_map = trajectory_step_map[:, prompt_length:]
        # assert (generated_trajectory_step_map >= 0).all(), 'Undecoded positions found: {}.'.format((generated_trajectory_step_map < 0).nonzero())

        return DLMGenerationOutput(sequences=x, trajectory_step_map=generated_trajectory_step_map)

    @torch.no_grad()
    def block_decode_with_block_attention(
        self, 
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        use_cache: bool = True,
        mask_token_id: int = 151669,
        eos_token_id: int = 151643,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
        early_exit: Optional[bool] = False,
        *args,
        **kwargs
    ):
        '''
        block_decode_with_block_causal_attention for block-causal models like SDAR and TraDo
        NOTE: now only consider 1 token per step; all the token transfer operations are performed in the block region.

        Returns:
            x: decoded ids (length padded to a multiple of ``block_length``).
            generated_trajectory_step_map: ``[batch, max_gen_length]`` step index when each of the first
            ``max_gen_length`` completion positions (after the original prompt) was first unmasked
            (``-1`` if still masked when decoding stops).
        '''

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'batch size must be 1 for block_decode_with_block_causal_attention'

        prompt_length = input_ids.shape[1]
        num_total_blocks = (prompt_length + max_gen_length + block_length - 1) // block_length # pad the whole sequence to be divisible by block_length
        max_length = num_total_blocks * block_length
        x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

        trajectory_step_map = torch.full((batch_size, max_length), -1, device=x.device, dtype=torch.long)
        global_step = 0

        num_prefill_blocks = prompt_length // block_length
        prefill_length = num_prefill_blocks * block_length
        
        # assert decoding_steps % num_total_blocks == 0, 'decoding_steps {} must be divisible by num_total_blocks {}'.format(decoding_steps, num_total_blocks)
        # steps_per_block = decoding_steps // num_total_blocks
        steps_per_block = math.ceil(block_length / self.token_per_step)

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_total_blocks,
            attention_type='block_causal'
        )
        past_key_values = DynamicCache()

        # Prefilling stage — use the [:prefill, :prefill] submatrix (same as trado_generate), not row index prefill_length.
        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_attn_mask = attention_mask[:, :, :prefill_length, :prefill_length]
            if cur_attn_mask.dim() == 3:
                cur_attn_mask = cur_attn_mask[:, None, :, :]
            cur_position_ids = position_ids[:, :prefill_length]
            self.get_processed_model_outputs(
                model=model,
                x=cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True
            )

        # Decoding stage
        for block_idx in range(num_prefill_blocks, num_total_blocks):

            current_block_start = block_idx * block_length
            current_block_end = current_block_start + block_length
            current_block_attention_mask = attention_mask[:, :, current_block_start:current_block_end, :current_block_end]
            if current_block_attention_mask.dim() == 3:
                current_block_attention_mask = current_block_attention_mask[:, None, :, :]
            current_block_position_ids = position_ids[:, current_block_start:current_block_end]
            current_block_x = x[:, current_block_start:current_block_end].clone()
            
            # decoding loop for the current block
            for step in range(steps_per_block):

                current_block_mask_index = (current_block_x == mask_token_id)

                logits, _ = self.get_processed_model_outputs(
                    model=model,
                    x=current_block_x,
                    attention_mask=current_block_attention_mask,
                    position_ids=current_block_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False
                )

                current_block_x0, current_block_confidence = self.get_mask_token_prediction_and_confidence(
                    logits=logits,
                    mask_index=current_block_mask_index,
                    mask_token_id=mask_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

                current_block_x = self.token_transfer(
                    x=current_block_x,
                    x0=current_block_x0,
                    confidence=current_block_confidence,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                    confidence_threshold=confidence_threshold
                )
                newly_unmasked = current_block_mask_index & (current_block_x != mask_token_id)
                if newly_unmasked.any():
                    trajectory_step_map[:, current_block_start:current_block_end][newly_unmasked] = global_step
                    global_step += 1

                if (current_block_x != mask_token_id).all():
                    # if the current block is all unmasked, store the current block's kv and exit the decoding loop
                    self.get_processed_model_outputs(
                        model=model,
                        x=current_block_x,
                        attention_mask=current_block_attention_mask,
                        position_ids=current_block_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True
                    )
                    break

            # commit the current block to the sequence
            x[:, current_block_start:current_block_end] = current_block_x

            # early exit if the current block is all eos tokens or pad tokens
            if early_exit and (x[:, current_block_start:current_block_end] == eos_token_id).all():
                if current_block_end < max_length:
                    x[:, current_block_end:] = eos_token_id
                    assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
                break

            # if early_exit and (x[:, current_block_start:current_block_end] == pad_token_id).all():
            #     if current_block_end < max_length:
            #         x[:, current_block_end:] = pad_token_id
            #         assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
            #     break

        generated_trajectory_step_map = trajectory_step_map[:, prompt_length:]
        # assert (generated_trajectory_step_map >= 0).all(), 'Undecoded positions found: {}.'.format((generated_trajectory_step_map < 0).nonzero())

        return DLMGenerationOutput(sequences=x, trajectory_step_map=generated_trajectory_step_map)

    @torch.no_grad()
    def block_decode_with_block_attention_FreeDave(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = 32,
        max_gen_length: Optional[int] = 128,
        decoding_steps: Optional[int] = 128,
        draft_steps: Optional[int] = 1,
        eager_acceptance_mode: Optional[bool] = False,
        draft_mode: Optional[str] = 'tree_attention',
        use_cache: bool = False,
        mask_token_id: int = 151669,
        eos_token_id: int = 151643,
        pad_token_id: int = 151643,
        pad_target_penalty: float = 1.0,
        confidence_threshold: Optional[float] = None,
        early_exit: Optional[bool] = False,
    ):

        '''
        FreeDave (https://arxiv.org/abs/2510.00294) for block-causal DLMs like SDAR and TraDo.
        NOTE: now only consider 1-token-per-step transition by default.

        Currently support two draft modes:
        - batch_expanding (v1): expand kv cache to match the batch_size of draft blocks, which requires more memory overhead but supports flash attention
        - tree_attention (v2): use tree attention (https://arxiv.org/abs/2401.10774) instead of expanding kv cache to further reduce memory overhead, but flash attention is not supported due to the non-null attention mask (https://github.com/pytorch/pytorch/blob/9594ae448e70a3503c5a483369f803810d20a5be/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L259)

        Although batch_expanding supports flash attention, tree_attention has lower memory overhead without frequent cache copy and reduction operations, and is empirically a bit faster.
        '''

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'batch size must be 1 for FreeDave'

        prompt_length = input_ids.shape[1]
        num_total_blocks = (prompt_length + max_gen_length + block_length - 1) // block_length # pad the whole sequence to be divisible by block_length
        max_length = num_total_blocks * block_length
        x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

        trajectory_step_map = torch.full((batch_size, max_length), -1, device=x.device, dtype=torch.long)
        hit_rates = []

        num_prefill_blocks = prompt_length // block_length
        prefill_length = num_prefill_blocks * block_length
        
        assert block_length % self.token_per_step == 0, 'block_length={} must be divisible by self.token_per_step={}'.format(block_length, self.token_per_step)
        steps_per_block = block_length // self.token_per_step

        attention_mask, position_ids = self.get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_total_blocks,
            attention_type='block_causal'
        )
        past_key_values = DynamicCache()

        # Prefilling stage — use the [:prefill, :prefill] submatrix, not row index prefill_length.
        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_attn_mask = attention_mask[..., :prefill_length, :prefill_length]
            if cur_attn_mask.dim() == 3:
                cur_attn_mask = cur_attn_mask[:, None, :, :]
            cur_position_ids = position_ids[:, :prefill_length]
            self.get_processed_model_outputs(
                model=model,
                x=cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )

        # Decoding stage
        draft_steps_list = []
        global_step = 0
        for block_idx in range(num_prefill_blocks, num_total_blocks):

            # determine the number of pre-allocated future blocks
            if eager_acceptance_mode:
                num_future_blocks = min(math.ceil(draft_steps / steps_per_block), num_total_blocks - block_idx - 1)
                confidence_priority_shift = torch.arange(num_future_blocks, -1, -1, device=x.device).repeat_interleave(block_length)
            else:
                num_future_blocks = 0
                confidence_priority_shift = None


            # TODO: dynamic allocate future blocks instead of static allocation; allocate only when the number of masked tokens in the current block is smaller than the draft steps
            current_block_start = block_idx * block_length
            current_block_end = current_block_start + block_length

            current_window_start = current_block_start
            current_window_end = current_window_start + (num_future_blocks + 1) * block_length
            current_window_len = current_window_end - current_window_start

            # NOTE: still use the block-causal attention mask, instead of just merge the current block and future blocks into a larger block with full attention mask
            current_window_attention_mask = attention_mask[..., current_window_start:current_window_end, :current_window_end] # (1, 1, current_window_len, current_window_end)
            if current_window_attention_mask.dim() == 3:
                current_window_attention_mask = current_window_attention_mask[:, None, :, :]
            current_window_position_ids = position_ids[:, current_window_start:current_window_end]

            current_window_x = x[:, current_window_start:current_window_end].clone() # (batch, current_window_len)
            current_window_mask_index = (current_window_x == mask_token_id)

            if current_window_mask_index[:, :block_length].sum() == 0:
                # if the current block is all unmasked, store the current block's kv and proceed to the next block
                self.get_processed_model_outputs(
                    model=model,
                    x=current_window_x[:, :block_length],
                    attention_mask=current_window_attention_mask[..., :block_length, :current_window_start + block_length],
                    position_ids=current_window_position_ids[:, :block_length],
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True
                )
                continue

            # first forward pass for the current block
            logits, _ = self.get_processed_model_outputs(
                model=model,
                x=current_window_x,
                attention_mask=current_window_attention_mask,
                position_ids=current_window_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False
            )

            current_window_x0, current_window_confidence = self.get_mask_token_prediction_and_confidence(
                logits=logits,
                mask_index=current_window_mask_index,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            # decoding loop for the current block
            while True:

                current_window_draft_steps = min(draft_steps, math.ceil(current_window_mask_index.sum().item() / self.token_per_step))

                current_window_x_draft = self.token_transfer(
                    x=current_window_x,
                    x0=current_window_x0,
                    draft_steps=current_window_draft_steps,
                    confidence=current_window_confidence,
                    confidence_priority_shift=confidence_priority_shift,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                ) # (batch_size * current_block_draft_steps, current_window_len)
                assert current_window_x_draft.shape == (batch_size * current_window_draft_steps, current_window_len), 'current_window_x_draft.shape must be (batch_size * current_window_draft_steps, current_window_len)'

                # if the last step in the last block, accept draft tokens and exit
                if num_total_blocks - block_idx == 1 and current_window_mask_index[:, :block_length].sum() <= self.token_per_step:
                    # here draft tokens should be sampled by single step
                    # Take every interval-th (current_block_draft_steps-th) sample in batch as per interval batch index
                    current_window_x = current_window_x_draft[::current_window_draft_steps]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        draft_steps_list.append(current_window_draft_steps)
                        global_step += 1
                    break

                # if eager mode disabled and the last step in the current block, accept draft tokens and exit
                if not eager_acceptance_mode and current_window_mask_index[:, :block_length].sum() <= self.token_per_step:
                    current_window_x = current_window_x_draft[::current_window_draft_steps]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        draft_steps_list.append(current_window_draft_steps)
                        global_step += 1
                    break

                # batch forward the draft blocks
                if draft_mode == 'batch_expanding':
                    past_key_values.batch_repeat_interleave(current_window_draft_steps)
                    logits, _ = self.get_processed_model_outputs(
                        model=model,
                        x=current_window_x_draft, # (batch_size * current_window_draft_steps, current_window_len)
                        attention_mask=current_window_attention_mask,
                        position_ids=current_window_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=False
                    ) # (batch_size * current_window_draft_steps, current_window_len, vocab_size)
                    past_key_values.batch_select_indices(torch.arange(0, current_window_x_draft.shape[0], current_window_draft_steps))

                    current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                        logits=logits, # (batch_size * current_window_draft_steps, current_window_len, vocab_size)
                        mask_index=(current_window_x_draft == mask_token_id), # (batch_size * current_window_draft_steps, current_window_len)
                        mask_token_id=mask_token_id,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )

                elif draft_mode == 'tree_attention':
                    current_window_draft_blocks_tree_attention_mask, current_window_draft_blocks_tree_position_ids = self.create_tree_attention_mask_and_position_ids(
                        attention_mask=current_window_attention_mask,
                        position_ids=current_window_position_ids,
                        shared_prefix_length=current_window_start,
                        branch_length=current_window_len,
                        shared_suffix_length=0,
                        num_branches=current_window_draft_steps
                    )

                    logits, _ = self.get_processed_model_outputs(
                        model=model,
                        x=current_window_x_draft.view(batch_size, -1), # (batch_size, current_window_len * current_window_draft_steps)
                        attention_mask=current_window_draft_blocks_tree_attention_mask,
                        position_ids=current_window_draft_blocks_tree_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=False
                    ) # (batch_size, current_window_len * current_window_draft_steps, vocab_size)

                    current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                        logits=logits.view(current_window_x_draft.shape[0], -1, logits.shape[-1]), # (batch_size * current_window_draft_steps, current_window_len, vocab_size)
                        mask_index=(current_window_x_draft == mask_token_id), # (batch_size * current_window_draft_steps, current_window_len)
                        mask_token_id=mask_token_id,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                else:
                    raise ValueError(f'Invalid draft mode: {draft_mode}. Please choose from "batch_expanding" or "tree_attention".')

                current_window_x_target = self.token_transfer(
                    x=current_window_x_draft,
                    x0=current_window_x0_draft,
                    confidence=current_window_confidence_draft,
                    confidence_priority_shift=confidence_priority_shift,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                )
                
                # draft tokens verification
                current_window_x_draft = current_window_x_draft.view(current_window_x.shape[0], current_window_draft_steps, *current_window_x.shape[1:]) # [batch_size, current_window_draft_steps, current_window_len]
                current_window_x_target = current_window_x_target.view(current_window_x.shape[0], current_window_draft_steps, *current_window_x.shape[1:]) # [batch_size, current_window_draft_steps, current_window_len]
                matched_draft_index = (current_window_x_target[:, :-1, :] == current_window_x_draft[:, 1:, :]).all(dim=-1)
                matched_steps = torch.cumprod(matched_draft_index, dim=-1).sum(dim=-1).item() # assume batch size is 1

                # Update current x, current kv cache, and reuse intermediate results
                # total_accepted_steps += matched_steps
                # total_draft_steps += current_window_draft_steps

                if eager_acceptance_mode and (current_window_x_draft[:, matched_steps, :block_length] != mask_token_id).all():
                    current_window_x = current_window_x_target[:, matched_steps, :]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        draft_steps_list.append(current_window_draft_steps)
                        global_step += 1
                else:
                    current_window_x = current_window_x_draft[:, matched_steps, :]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        draft_steps_list.append(current_window_draft_steps)
                        global_step += 1
                    current_window_x0 = current_window_x0_draft.view(batch_size, current_window_draft_steps, *current_window_x.shape[1:])[:, matched_steps, :]
                    current_window_confidence = current_window_confidence_draft.view(batch_size, current_window_draft_steps, *current_window_x.shape[1:])[:, matched_steps, :]

                if (current_window_x[:, :block_length] != mask_token_id).all():
                    # if the current block is all unmasked, exit the decoding loop
                    break
                else:
                    current_window_mask_index = (current_window_x == mask_token_id)

            # store the current block's kv
            self.get_processed_model_outputs(
                model=model,
                x=current_window_x[:, :block_length],
                attention_mask=current_window_attention_mask[..., :block_length, :current_window_start + block_length],
                position_ids=current_window_position_ids[:, :block_length],
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True
            )
            # commit the current block and future blocks (if any) to the sequence
            x[:, current_window_start:current_window_end] = current_window_x

            # early exit if the current block is all eos tokens or pad tokens
            if early_exit and (x[:, current_block_start:current_block_end] == eos_token_id).all():
                if current_block_end < max_length:
                    x[:, current_block_end:] = eos_token_id
                    assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
                break

            # if early_exit and (x[:, current_block_start:current_block_end] == pad_token_id).all():
            #     if current_block_end < max_length:
            #         x[:, current_block_end:] = pad_token_id
            #         assert (trajectory_step_map[:, prompt_length:current_block_end] >= 0).all(), 'Undecoded positions found: {}.'.format((trajectory_step_map[:, prompt_length:current_block_end] < 0).nonzero())
            #     break

        generated_trajectory_step_map = trajectory_step_map[:, prompt_length:]
        # assert (generated_trajectory_step_map >= 0).all(), 'Undecoded positions found: {}.'.format((generated_trajectory_step_map < 0).nonzero())
        
        # for i in range(len(draft_steps_list)):
        #     hit_rate = (generated_trajectory_step_map == i).sum().item() / draft_steps_list[i]
        #     hit_rates.append(hit_rate)

        return DLMGenerationOutput(sequences=x, trajectory_step_map=generated_trajectory_step_map)