import torch
from typing import Optional, TYPE_CHECKING
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

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

def get_attention_mask_and_position_ids(
    attention_mask: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
    block_length: Optional[int] = None, 
    num_blocks: Optional[int] = None, 
    attention_type: Optional[str] = None,
    device: Optional[torch.device] = None
):
    '''
        Create 4D attention mask tensor and 2D position ids tensor for full attention or block-causal attention.
    '''
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
            attention_mask = torch.ones(1, 1, max_length, max_length, device=device)
            position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    elif attention_type == 'block_causal':
        if block_length is None or num_blocks is None:
            raise ValueError('block_length and num_blocks must be provided when attention_type is block_causal')
        inter_block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
        attention_mask = inter_block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0).unsqueeze(0) # (1, 1, num_blocks * block_length, num_blocks * block_length)
        position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    else:
        logger.warning(f'Attention type {attention_type} not supported. Fall back to full attention.')
        attention_mask = torch.ones(1, 1, max_length, max_length, device=device)
        position_ids = torch.arange(max_length, device=device).unsqueeze(0)

    assert attention_mask.ndim == 4, 'attention_mask must be 4D, but got {attention_mask.shape}'
    assert position_ids.ndim == 2, 'position_ids must be 2D, but got {position_ids.shape}'

    return attention_mask, position_ids

def get_tree_attention_mask_and_position_ids(
    attention_mask: Optional[torch.Tensor] = None, 
    position_ids: Optional[torch.Tensor] = None, 
    shared_prefix_length: Optional[int] = None,
    branch_length: Optional[int] = None,
    shared_suffix_length: Optional[int] = 0,
    num_branches: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    '''
    Create tree-structured attention mask (4D, 0/1) and position ids for draft-tree verification.
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
            device=device,
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
        position_ids = torch.arange(shared_prefix_length, shared_prefix_length + branch_length, device=device).unsqueeze(0)
    elif position_ids.ndim == 2 and position_ids.shape[1] == expected_width:
        # position_ids may include [prefix || branch || suffix]; keep branch span only.
        position_ids = position_ids[:, shared_prefix_length:shared_prefix_length + branch_length]
    elif position_ids.ndim == 2 and position_ids.shape[1] != branch_length:
        raise ValueError(
            f"position_ids shape mismatch: got {tuple(position_ids.shape)}, "
            f"expected second dim {branch_length} or {expected_width}."
        )

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