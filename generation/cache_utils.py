from transformers.cache_utils import DynamicCache
import torch
from typing import Optional, Dict, Any, Tuple


class DynamicDualCache(DynamicCache):
    """
    Dual cache that keeps a frozen base KV cache and composes:
    [prefix-from-base, current-forward-kv, suffix-from-base]
    on each update.

    This avoids storing/rewriting the mutable window in the cache and works
    for both block decoding and tree-attention verification.
    """

    def __init__(self):
        super().__init__()
        self.prefix_end: Optional[int] = None
        self.suffix_start: Optional[int] = None

    def set_dual_layout(self, prefix_end: int, suffix_start: int):
        prefix_end = int(prefix_end)
        suffix_start = int(suffix_start)
        if prefix_end < 0:
            raise ValueError(f"prefix_end must be >= 0, got {prefix_end}.")
        if suffix_start < 0:
            raise ValueError(f"suffix_start must be >= 0, got {suffix_start}.")
        if suffix_start < prefix_end:
            raise ValueError(
                "Invalid dual layout: suffix_start must be >= prefix_end, got "
                f"suffix_start={suffix_start}, prefix_end={prefix_end}."
            )
        self.prefix_end = prefix_end
        self.suffix_start = suffix_start

    def clear_dual_layout(self):
        self.prefix_end = None
        self.suffix_start = None

    def _store_base_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif not self.key_cache[layer_idx].numel():
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if layer_idx == 0 and len(self.key_cache) <= layer_idx:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx or not self.key_cache[layer_idx].numel():
            self._store_base_layer(key_states, value_states, layer_idx)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if self.prefix_end is None or self.suffix_start is None:
            # Fallback to standard DynamicCache behavior when no dual layout set.
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        base_key = self.key_cache[layer_idx]
        base_value = self.value_cache[layer_idx]
        base_seq_len = base_key.shape[-2]

        if self.prefix_end > base_seq_len or self.suffix_start > base_seq_len:
            raise ValueError(
                "Dual layout out of range: "
                f"prefix_end={self.prefix_end}, "
                f"suffix_start={self.suffix_start}, base_seq_len={base_seq_len}."
            )
        if self.prefix_end > self.suffix_start:
            raise ValueError(
                "Invalid dual layout: prefix_end must be <= suffix_start, got "
                f"{self.prefix_end} > {self.suffix_start}."
            )

        prefix_key = base_key[:, :, : self.prefix_end, :]
        prefix_value = base_value[:, :, : self.prefix_end, :]
        suffix_key = base_key[:, :, self.suffix_start :, :]
        suffix_value = base_value[:, :, self.suffix_start :, :]
        out_key = torch.cat([prefix_key, key_states, suffix_key], dim=-2)
        out_value = torch.cat([prefix_value, value_states, suffix_value], dim=-2)
        return out_key, out_value
