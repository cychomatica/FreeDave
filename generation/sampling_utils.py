import torch
import torch.nn.functional as F

def top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(
            logits < min_values, torch.full_like(logits, float("-inf")), logits
        )

def top_p_logits(logits, p):
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1,
        sorted_indices,
        sorted_mask,
    )
    return logits.masked_fill(mask_indices, float("-inf"))

def sample_tokens(logits, temperature=1.0, top_k=None, top_p=None):
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    if temperature == 0.0:
        token = torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)