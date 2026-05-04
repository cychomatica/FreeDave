import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.utils import logging
from typing import Optional, List, Union
from deprecated import deprecated
from .sampling_utils import sample_tokens
import math
from transformers.cache_utils import DynamicCache
from .cache_utils import DynamicDualCache
from .attn_utils import (
    visibility_mask_to_sdpa_additive,
    get_attention_mask_and_position_ids,
    get_tree_attention_mask_and_position_ids,
)
from .monitor_utils import debug_only, ForwardMonitor
from dataclasses import dataclass

from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.get_logger(__name__)

# Currently only include Dream models. Extend this list if using other AR-adapted DLMs.
AR_ADAPTED_DLM = ['DreamModel', 'DreamBaseModel']

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
    sequences: Optional[torch.Tensor] = None
    trajectory_step_map: Optional[torch.Tensor] = None
    num_proposed_tokens: Optional[torch.Tensor] = None
    num_hit_tokens: Optional[torch.Tensor] = None

class DLMGeneration:

    def __init__(
        self,
        sdpa_additive_attention_mask: bool = False,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            sdpa_additive_attention_mask: If True, 4D attention masks are converted from visibility (1/0 or bool) to SDPA additive form before each model forward. Use for backbones that do not convert masks internally (e.g. LLaDA/Dream SDPA). Keep False for TraDo/SDAR, which expect 1/0 and convert inside `SDARAttention`. 
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_per_step = 1
        self.sdpa_additive_attention_mask = sdpa_additive_attention_mask
        self.seed = seed
        self.spda_backend = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH
        ]

    def _prepare_attention_mask_for_forward(
        self,
        model: nn.Module,
        attention_mask: Optional[torch.Tensor],
    ):
        if attention_mask is None:
            return None
            
        if self.sdpa_additive_attention_mask:
            try:
                w_dtype = next(model.parameters()).dtype
            except StopIteration:
                w_dtype = torch.float32
            return visibility_mask_to_sdpa_additive(attention_mask, w_dtype)

        return attention_mask

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

        attention_mask = self._prepare_attention_mask_for_forward(model, attention_mask)

        if is_ar_adapted_dlm_model(model):
            with sdpa_kernel(self.spda_backend):
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
            with sdpa_kernel(self.spda_backend):
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


    @deprecated('This function is deprecated, now consider 1 token per step by default')
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
        confidence[~mask_index] = -torch.inf # set the confidence of unmasked tokens to -inf

        if confidence_threshold is not None:
            # confidence-aware parallel decoding
            x_draft = x.clone()

            if confidence_priority_shift is None:
                high_confidence_index = (confidence >= confidence_threshold)
            else:
                confidence_priority_shift = mask_index.int() * confidence_priority_shift
                confidence = confidence + confidence_priority_shift
                # Per-row max shift (do not use .max() without dim on [batch, seq] — that would mix batches).
                shift_row_max = confidence_priority_shift.max(dim=-1, keepdim=True).values
                high_confidence_index = (confidence >= confidence_threshold + shift_row_max)

            transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)
            
            if alg_temp is None or alg_temp == 0:
                for j in range(confidence.shape[0]):
                    if high_confidence_index[j].any():
                        transfer_index[j, high_confidence_index[j]] = True
                    else:
                        _, idx = torch.topk(confidence[j], min(token_per_step, mask_index[j].sum()))
                        transfer_index[j, idx] = True
                x_draft[transfer_index] = x0[transfer_index]
            elif alg_temp == -1: # find the left-to-right consecutive tokens with highest confidence; if none, unmask the leftmost token
                for j in range(confidence.shape[0]):
                    mask_high_confidence_index = high_confidence_index[j, mask_index[j]]
                    if mask_high_confidence_index.cumprod(dim=-1).sum().item() > 0:
                        idx = torch.arange(x.size(1), device=self.device)[mask_index[j]][:mask_high_confidence_index.cumprod(dim=-1).sum().item()]
                    else:
                        idx = torch.arange(x.size(1), device=self.device)[mask_index[j]][:token_per_step]
                    transfer_index[j, idx] = True
                x_draft[transfer_index] = x0[transfer_index]
            else:
                raise NotImplementedError(f'Algorithm temperature {alg_temp} not supported yet.')

            if draft_steps is None or draft_steps == 1:
                return x_draft
            else:
                # FreeDave++: additional draft based on confidence-aware parallel decoding
                if confidence_priority_shift is not None:
                    confidence = confidence - confidence_priority_shift
                x_draft_additional = self.token_transfer(
                    x=x_draft,
                    x0=x0,
                    confidence=confidence,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    confidence_priority_shift=confidence_priority_shift,
                    alg_temp=alg_temp,
                    confidence_threshold=None,
                    draft_steps=draft_steps-1,
                )
                return torch.cat([x_draft, x_draft_additional], dim=0)

        else:
            if draft_steps is not None and draft_steps > 1:
                x_draft = x.unsqueeze(1).expand(x.shape[0], draft_steps, *x.shape[1:]).clone() # (batch, draft_steps, seq_len)
            else:
                draft_steps = 1
                x_draft = x.unsqueeze(1).clone() # (batch, 1, seq_len)
            transfer_index = torch.zeros_like(x_draft, dtype=torch.bool)

            if confidence_priority_shift is not None:
                confidence = confidence + confidence_priority_shift # apply left-to-right block priority shift

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

    def draft_verification(
        self,
        x_draft: torch.Tensor,
        x_target: torch.Tensor,
        mask_token_id: int,
        match_rule: Optional[str] = 'exact_match',
    ):
        return self.draft_verification_vectorized(x_draft, x_target, mask_token_id, match_rule)

    def draft_verification_vectorized(
        self,
        x_draft: torch.Tensor,
        x_target: torch.Tensor,
        mask_token_id: int,
        match_rule: Optional[str] = 'exact_match',
    ):
        """
        Greedy chain by per-row agreement, vectorized.

        Equivalent to the reference Python loop in ``_draft_verification_loop``:
        starting from ``best = 0``, repeatedly take the LARGEST ``j > best`` such that
        ``draft[j]`` agrees with ``target[best]`` (subset rule on unmasked positions,
        or full equality), set ``best = j``; stop at a fixed point.

        Implementation: precompute the ``[B, K, K]`` agreement matrix once
        (``A[b, i, j] = agrees(draft[b, j], target[b, i])``), restrict to the strict
        upper triangle, then run ``K - 1`` GPU-side gather/argmax steps. No Python loop
        over batch / draft_steps and no ``.item()`` / ``.any()`` syncs.
        """
        batch_size, num_draft_branches, _ = x_draft.shape
        if x_target.shape != x_draft.shape:
            raise ValueError(
                f"x_target {x_target.shape} and x_draft {x_draft.shape} must match"
            )
        if num_draft_branches == 1:
            return torch.zeros(batch_size, dtype=torch.long, device=x_draft.device)

        device = x_draft.device
        K = num_draft_branches

        draft_b1kl = x_draft.unsqueeze(1)   # [B, 1, K, L]  -> "j"
        target_bk1l = x_target.unsqueeze(2) # [B, K, 1, L]  -> "i"
        if match_rule == 'exact_match':
            per_pos = draft_b1kl == target_bk1l
        elif match_rule == 'subset':
            per_pos = (draft_b1kl == target_bk1l) | (draft_b1kl == mask_token_id)
        else:
            raise NotImplementedError(f'Rule {match_rule} not supported yet.')
        agree = per_pos.all(dim=-1)  # [B, K, K]; agree[b, i, j] = agrees(draft[b, j], target[b, i])
        # Reference rule: an all-mask draft is unconditionally agreed under any match_rule.
        draft_all_mask = (x_draft == mask_token_id).all(dim=-1)  # [B, K], indexed by "j"
        agree = agree | draft_all_mask.unsqueeze(1).expand(batch_size, K, K)
        triu = torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)
        agree = agree & triu  # only j > i is a valid advance

        arange_K = torch.arange(K, device=device).unsqueeze(0).expand(batch_size, K)  # [B, K]
        neg_one = torch.full((batch_size, K), -1, dtype=torch.long, device=device)
        best = torch.zeros(batch_size, dtype=torch.long, device=device)
        for _ in range(K - 1):  # at most K-1 advances; further passes are no-ops at fixed point
            row = agree.gather(1, best.view(batch_size, 1, 1).expand(batch_size, 1, K)).squeeze(1)  # [B, K]
            candidates = torch.where(row, arange_K, neg_one)
            new_best = candidates.max(dim=-1).values  # largest valid j, or -1 if none
            best = torch.where(new_best > best, new_best, best)
        return best

    def draft_verification_looped(
        self,
        x_draft: torch.Tensor,
        x_target: torch.Tensor,
        mask_token_id: int,
        match_rule: Optional[str] = 'exact_match',
    ):
        """
        Greedy chain by *subset* agreement on unmasked positions.

        Start at branch ``0``. Each outer pass fixes ``ref_target_idx = best_draft_idx``,
        then scans ``draft_idx = best_draft_idx + 1, ..., num_draft_branches - 1``;
        whenever ``draft[draft_idx]`` is a subset of ``target[ref_target_idx]`` (see
        below), set ``best_draft_idx = draft_idx`` (later hits in the scan win). If a
        full pass leaves ``best_draft_idx`` unchanged, stop.

        Subset: every position where ``draft`` is not ``mask_token_id`` is unmasked in
        ``ref_target`` with the same token id.

        Returns ``best_draft_idx`` per batch row.
        """
        batch_size, num_draft_branches, _ = x_draft.shape
        if x_target.shape != x_draft.shape:
            raise ValueError(
                f"x_target {x_target.shape} and x_draft {x_draft.shape} must match"
            )
        device = x_draft.device
        accepted_branch_index = torch.zeros(batch_size, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            
            best_draft_idx = 0
            while best_draft_idx < num_draft_branches - 1:

                ref_target_idx = best_draft_idx

                for draft_idx in range(best_draft_idx + 1, num_draft_branches):

                    draft = x_draft[batch_idx, draft_idx]
                    ref_target = x_target[batch_idx, ref_target_idx]
                    draft_unmasked_positions = draft != mask_token_id

                    if not draft_unmasked_positions.any():
                        draft_match_ref_target = True
                    else:
                        if match_rule == 'exact_match':
                            draft_match_ref_target = (draft == ref_target).all()
                        elif match_rule == 'subset':
                            draft_match_ref_target = (draft[draft_unmasked_positions] == ref_target[draft_unmasked_positions]).all()
                        else:
                            raise NotImplementedError(f'Rule {match_rule} not supported yet.')
                    if not draft_match_ref_target:
                        continue

                    best_draft_idx = draft_idx

                if best_draft_idx == ref_target_idx:
                    break

            accepted_branch_index[batch_idx] = best_draft_idx

        return accepted_branch_index

    @debug_only(message="Calling DLMGeneration._freedave_debug_sequential_draft_forward(). This will forward each draft branch sequentially instead of in parallel. Make sure it is for determinism debugging only.")
    def _freedave_debug_sequential_draft_forward(
        self,
        model: nn.Module,
        current_window_x_draft: torch.Tensor,
        current_window_attention_mask: Optional[torch.Tensor],
        current_window_position_ids: Optional[torch.Tensor],
        past_key_values: Union[DynamicCache, DynamicDualCache],
        current_window_draft_steps: int,
        suffix_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Debug-only: one forward per draft path with batch=1, same mask/position_ids layout as
        the corresponding baseline draft forward. Clones ``past_key_values`` each iteration
        (required for Dream; TraDo with store_kv=False does not mutate cache, but cloning
        keeps behavior uniform). If ``suffix_ids`` is set (non-dual full attention), concat
        ``[window || suffix]`` like the batch_expanding path.
        """
        def clone_cache_for_debug(cache: Union[DynamicCache, DynamicDualCache]):
            """Clone KV (and dual-layout metadata) so each verify forward starts from the same prefix state."""
            if isinstance(cache, DynamicDualCache):
                out = DynamicDualCache()
                for k, v in zip(cache.key_cache, cache.value_cache):
                    out.key_cache.append(k.clone())
                    out.value_cache.append(v.clone())
                if hasattr(cache, "_seen_tokens"):
                    out._seen_tokens = cache._seen_tokens
                if cache.prefix_end is not None:
                    out.prefix_end = cache.prefix_end
                    out.suffix_start = cache.suffix_start
                return out
            out = DynamicCache()
            for k, v in zip(cache.key_cache, cache.value_cache):
                out.key_cache.append(k.clone())
                out.value_cache.append(v.clone())
            if hasattr(cache, "_seen_tokens"):
                out._seen_tokens = cache._seen_tokens
            return out

        logits_chunks: List[torch.Tensor] = []
        for k in range(current_window_draft_steps):
            pkv = clone_cache_for_debug(past_key_values)
            x_in = current_window_x_draft[k : k + 1]
            if suffix_ids is not None:
                x_in = torch.cat([x_in, suffix_ids], dim=-1)
            logits_k, _ = self.get_processed_model_outputs(
                model=model,
                x=x_in,
                attention_mask=current_window_attention_mask,
                position_ids=current_window_position_ids,
                past_key_values=pkv,
                use_cache=True,
                store_kv=False,
            )
            logits_chunks.append(logits_k)
        logits = torch.cat(logits_chunks, dim=0)

        # NFE correction: this debug path turns a single batch forward into `current_window_draft_steps` sequential forwards. When ForwardMonitor is active (i.e. inside `with monitor.count():`), merge these into a single effective forward for decoding stats correction.
        fm = ForwardMonitor.get_active()
        if fm is not None and current_window_draft_steps > 1:
            fm.nfe = max(0, int(fm.nfe) - (current_window_draft_steps - 1))

        return logits
    
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

        attention_mask, position_ids = get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_gen_blocks,
            attention_type='full',
            device=self.device
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

            # Prepare attention mask and position ids for current block
            if dual_cache:
                current_block_attention_mask = attention_mask[:, :, current_block_start:current_block_end, :]
                current_block_position_ids = position_ids[:, current_block_start:current_block_end]
            else:
                current_block_attention_mask = attention_mask[:, :, current_block_start:, :]
                current_block_position_ids = position_ids[:, current_block_start:]

            # decoding loop for the current block
            step = 1
            while step < steps_per_block:

                current_block_mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                
                if dual_cache:
                    logits, _ = self.get_processed_model_outputs(
                        model=model,
                        x=x[:, current_block_start:current_block_end], # use the current block for forward pass
                        attention_mask=current_block_attention_mask, 
                        position_ids=current_block_position_ids, 
                        past_key_values=past_key_values, 
                        use_cache=use_cache
                    )
                else:
                    logits, _ = self.get_processed_model_outputs(
                        model=model,
                        x=x[:, current_block_start:], # use the entire remaining sequence for forward pass
                        attention_mask=current_block_attention_mask, 
                        position_ids=current_block_position_ids, 
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

        Draft modes: ``batch_expanding``, ``tree_attention``, or ``debug`` (sequential one-forward-per-draft;
        slow; isolates algorithm vs. kernel differences — see ``_freedave_debug_sequential_draft_forward``).
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

        attention_mask, position_ids = get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_gen_blocks,
            attention_type='full',
            device=self.device
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
            else: # Extract only the prefix cache from previous blocks
                past_key_values.crop(current_block_start)

            # Prepare attention mask and position ids for the current window
            if dual_cache:
                current_window_attention_mask = attention_mask[:, :, current_block_start:current_block_end, :]
                current_window_position_ids = position_ids[:, current_block_start:current_block_end]
            else:
                current_window_attention_mask = attention_mask[:, :, current_block_start:, :]
                current_window_position_ids = position_ids[:, current_block_start:]

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
                        confidence_threshold=confidence_threshold,
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
                            attention_mask=current_window_attention_mask,
                            position_ids=current_window_position_ids,
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
                        tree_attention_mask, tree_position_ids = get_tree_attention_mask_and_position_ids(
                            attention_mask=current_window_attention_mask,
                            position_ids=current_window_position_ids,
                            shared_prefix_length=current_window_start,
                            branch_length=current_window_len,
                            shared_suffix_length=max_length-current_window_end,
                            num_branches=current_window_draft_steps,
                            device=self.device
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
                    elif draft_mode == 'debug':
                        logits = self._freedave_debug_sequential_draft_forward(
                            model=model,
                            current_window_x_draft=current_window_x_draft,
                            current_window_attention_mask=None,
                            current_window_position_ids=torch.arange(
                                current_window_start, current_window_end, device=x.device
                            ).unsqueeze(0),
                            past_key_values=past_key_values,
                            current_window_draft_steps=current_window_draft_steps,
                        )
                        current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits,
                            mask_index=(current_window_x_draft == mask_token_id),
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    else:
                        raise ValueError(
                            f'Invalid draft mode: {draft_mode}. '
                            'Choose from "batch_expanding", "tree_attention", or "debug".'
                        )
                else:
                    if draft_mode == 'batch_expanding':
                        past_key_values.batch_repeat_interleave(current_window_draft_steps)
                        logits, _ = self.get_processed_model_outputs(
                            model=model,
                            x=torch.cat([current_window_x_draft, x[:, current_window_end:].expand(current_window_x_draft.shape[0], -1)], dim=-1),
                            attention_mask=current_window_attention_mask,
                            position_ids=current_window_position_ids,
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
                        tree_attention_mask, tree_position_ids = get_tree_attention_mask_and_position_ids(
                            attention_mask=current_window_attention_mask,
                            position_ids=current_window_position_ids,
                            shared_prefix_length=current_window_start,
                            branch_length=max_length-current_window_start,
                            shared_suffix_length=0,
                            num_branches=current_window_draft_steps,
                            device=self.device
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

                    elif draft_mode == 'debug':
                        logits = self._freedave_debug_sequential_draft_forward(
                            model=model,
                            current_window_x_draft=current_window_x_draft,
                            current_window_attention_mask=current_window_attention_mask,
                            current_window_position_ids=current_window_position_ids,
                            past_key_values=past_key_values,
                            current_window_draft_steps=current_window_draft_steps,
                            suffix_ids=x[:, current_window_end:],
                        )
                        current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                            logits=logits[:, :current_window_len, :],
                            mask_index=(current_window_x_draft == mask_token_id),
                            mask_token_id=mask_token_id,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    else:
                        raise ValueError(
                            f'Invalid draft mode: {draft_mode}. '
                            'Choose from "batch_expanding", "tree_attention", or "debug".'
                        )

                step += 1

                current_window_x_target = self.token_transfer(
                    x=current_window_x_draft,
                    x0=current_window_x0_draft,
                    confidence=current_window_confidence_draft,
                    confidence_priority_shift=confidence_priority_shift,
                    confidence_threshold=confidence_threshold,
                    draft_steps=None,
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
                accepted_branch_index = self.draft_verification(
                    current_window_x_draft,
                    current_window_x_target,
                    mask_token_id,
                    match_rule='subset',
                )
                batch_row_indices = torch.arange(batch_size, device=current_window_x_draft.device)

                if eager_acceptance_mode and (current_window_x_draft[batch_row_indices, accepted_branch_index, :block_length] != mask_token_id).all():
                    x[:, current_window_start:current_window_end] = current_window_x_target[batch_row_indices, accepted_branch_index, :]
                    newly_unmasked = current_window_mask_index & (x[:, current_window_start:current_window_end] != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                else:
                    x[:, current_window_start:current_window_end] = current_window_x_draft[batch_row_indices, accepted_branch_index, :]
                    newly_unmasked = current_window_mask_index & (x[:, current_window_start:current_window_end] != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                    current_window_x0 = current_window_x0_draft.view(
                        batch_size, current_window_draft_steps, current_window_len
                    )[batch_row_indices, accepted_branch_index, :]
                    current_window_confidence = current_window_confidence_draft.view(
                        batch_size, current_window_draft_steps, current_window_len
                    )[batch_row_indices, accepted_branch_index, :]

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
        num_proposed_tokens, num_hit_tokens = 0, 0
        global_step = 0

        num_prefill_blocks = prompt_length // block_length
        prefill_length = num_prefill_blocks * block_length
        
        # assert decoding_steps % num_total_blocks == 0, 'decoding_steps {} must be divisible by num_total_blocks {}'.format(decoding_steps, num_total_blocks)
        # steps_per_block = decoding_steps // num_total_blocks
        steps_per_block = math.ceil(block_length / self.token_per_step)

        attention_mask, position_ids = get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_total_blocks,
            attention_type='block_causal',
            device=self.device
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
                    num_proposed_tokens += newly_unmasked.sum()
                    num_hit_tokens += newly_unmasked.sum()

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

        return DLMGenerationOutput(
            sequences=x, 
            trajectory_step_map=generated_trajectory_step_map, 
            num_proposed_tokens=num_proposed_tokens, 
            num_hit_tokens=num_hit_tokens
        )

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

        Supported draft modes:
        - batch_expanding (v1): expand kv cache to match the batch_size of draft blocks, which requires more memory overhead but supports flash attention
        - tree_attention (v2): use tree attention (https://arxiv.org/abs/2401.10774) instead of expanding kv cache to further reduce memory overhead, but flash attention is not supported due to the non-null attention mask (https://github.com/pytorch/pytorch/blob/9594ae448e70a3503c5a483369f803810d20a5be/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L259)
        - debug: one forward per draft path with batch=1, same block-causal mask as baseline; clones the cache each time so KV layout matches sequential decoding (slow; for isolating algorithm bugs vs. kernel / accumulation order)

        Although batch_expanding supports flash attention, tree_attention has lower memory overhead without frequent cache copy and reduction operations, and is empirically a bit faster.
        '''

        batch_size = input_ids.shape[0]
        assert batch_size == 1, 'batch size must be 1 for FreeDave'

        prompt_length = input_ids.shape[1]
        num_total_blocks = (prompt_length + max_gen_length + block_length - 1) // block_length # pad the whole sequence to be divisible by block_length
        max_length = num_total_blocks * block_length
        x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

        trajectory_step_map = torch.full((batch_size, max_length), -1, device=x.device, dtype=torch.long)
        num_proposed_tokens, num_hit_tokens = 0, 0

        num_prefill_blocks = prompt_length // block_length
        prefill_length = num_prefill_blocks * block_length
        
        assert block_length % self.token_per_step == 0, 'block_length={} must be divisible by self.token_per_step={}'.format(block_length, self.token_per_step)
        steps_per_block = block_length // self.token_per_step

        attention_mask, position_ids = get_attention_mask_and_position_ids(
            attention_mask=attention_mask,
            max_length=max_length,
            block_length=block_length,
            num_blocks=num_total_blocks,
            attention_type='block_causal',
            device=self.device
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
                    confidence_threshold=confidence_threshold,
                ) # (batch_size * current_block_draft_steps, current_window_len)
                assert current_window_x_draft.shape == (batch_size * current_window_draft_steps, current_window_len), 'current_window_x_draft.shape must be (batch_size * current_window_draft_steps, current_window_len)'

                num_proposed_tokens += (
                    current_window_mask_index & (current_window_x_draft[current_window_draft_steps-1::current_window_draft_steps] != mask_token_id)
                ).sum().item()

                # if the last step in the last block, accept draft tokens and exit
                if num_total_blocks - block_idx == 1 and current_window_mask_index[:, :block_length].sum() <= self.token_per_step:
                    # here draft tokens should be sampled by single step
                    # Take every interval-th (current_block_draft_steps-th) sample in batch as per interval batch index
                    current_window_x = current_window_x_draft[::current_window_draft_steps]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                        num_hit_tokens += newly_unmasked.sum()
                    break

                # if eager mode disabled and the last step in the current block, accept draft tokens and exit
                if not eager_acceptance_mode and current_window_mask_index[:, :block_length].sum() <= self.token_per_step:
                    current_window_x = current_window_x_draft[::current_window_draft_steps]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                        num_hit_tokens += newly_unmasked.sum()
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
                    current_window_draft_blocks_tree_attention_mask, current_window_draft_blocks_tree_position_ids = get_tree_attention_mask_and_position_ids(
                        attention_mask=current_window_attention_mask,
                        position_ids=current_window_position_ids,
                        shared_prefix_length=current_window_start,
                        branch_length=current_window_len,
                        shared_suffix_length=0,
                        num_branches=current_window_draft_steps,
                        device=self.device
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

                elif draft_mode == 'debug':
                    logits = self._freedave_debug_sequential_draft_forward(
                        model=model,
                        current_window_x_draft=current_window_x_draft,
                        current_window_attention_mask=current_window_attention_mask,
                        current_window_position_ids=current_window_position_ids,
                        past_key_values=past_key_values,
                        current_window_draft_steps=current_window_draft_steps,
                    )
                    current_window_x0_draft, current_window_confidence_draft = self.get_mask_token_prediction_and_confidence(
                        logits=logits,
                        mask_index=(current_window_x_draft == mask_token_id),
                        mask_token_id=mask_token_id,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )

                else:
                    raise ValueError(
                        f'Invalid draft mode: {draft_mode}. '
                        'Choose from "batch_expanding", "tree_attention", or "debug".'
                    )

                current_window_x_target = self.token_transfer(
                    x=current_window_x_draft,
                    x0=current_window_x0_draft,
                    confidence=current_window_confidence_draft,
                    confidence_priority_shift=confidence_priority_shift,
                    block_length=block_length,
                    mask_token_id=mask_token_id,
                    alg_temp=alg_temp,
                    token_per_step=self.token_per_step,
                    confidence_threshold=confidence_threshold,
                )
                
                # draft tokens verification
                current_window_x_draft = current_window_x_draft.view(current_window_x.shape[0], current_window_draft_steps, *current_window_x.shape[1:]) # [batch_size, current_window_draft_steps, current_window_len]
                current_window_x_target = current_window_x_target.view(current_window_x.shape[0], current_window_draft_steps, *current_window_x.shape[1:]) # [batch_size, current_window_draft_steps, current_window_len]
                accepted_branch_index = self.draft_verification(
                    current_window_x_draft,
                    current_window_x_target,
                    mask_token_id,
                )
                batch_row_indices = torch.arange(batch_size, device=current_window_x_draft.device)

                if eager_acceptance_mode and (current_window_x_draft[batch_row_indices, accepted_branch_index, :block_length] != mask_token_id).all():
                    current_window_x = current_window_x_target[batch_row_indices, accepted_branch_index, :]
                    num_proposed_tokens += (current_window_x != current_window_x_draft[batch_row_indices, accepted_branch_index, :]).sum() # correction for target commit under eager acceptance mode
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                        num_hit_tokens += newly_unmasked.sum()
                else:
                    current_window_x = current_window_x_draft[batch_row_indices, accepted_branch_index, :]
                    newly_unmasked = current_window_mask_index & (current_window_x != mask_token_id)
                    if newly_unmasked.any():
                        trajectory_step_map[:, current_window_start:current_window_end][newly_unmasked] = global_step
                        global_step += 1
                        num_hit_tokens += newly_unmasked.sum()
                    current_window_x0 = current_window_x0_draft.view(batch_size, current_window_draft_steps, *current_window_x.shape[1:])[batch_row_indices, accepted_branch_index, :]
                    current_window_confidence = current_window_confidence_draft.view(batch_size, current_window_draft_steps, *current_window_x.shape[1:])[batch_row_indices, accepted_branch_index, :]

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
        assert (generated_trajectory_step_map >= 0).sum().item() == num_hit_tokens, '(generated_trajectory_step_map >= 0).sum() must be equal to num_hit_tokens'
        # assert (generated_trajectory_step_map >= 0).all(), 'Undecoded positions found: {}.'.format((generated_trajectory_step_map < 0).nonzero())

        return DLMGenerationOutput(
            sequences=x, 
            trajectory_step_map=generated_trajectory_step_map, 
            num_proposed_tokens=num_proposed_tokens, 
            num_hit_tokens=num_hit_tokens
        )