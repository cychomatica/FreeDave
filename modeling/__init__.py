# The modeling files on huggingface do not directly support flex attention, so we wrap them locally.
# For each DLM, we basically copy the original modeling file and make some tweaks to support flex attention.
from transformers import AutoTokenizer

def get_model(model_name: str, **kwargs):
    if model_name in (
        "Dream-org/Dream-v0-Instruct-7B",
        "Dream-org/Dream-v0-Base-7B",
    ):
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_name, **kwargs)
    elif model_name in (
        "Gen-Verse/TraDo-4B-Instruct",
        "Gen-Verse/TraDo-8B-Instruct",
        "Gen-Verse/TraDo-8B-Thinking"
    ):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif model_name in (
        "GSAI-ML/LLaDA-8B-Instruct",
        "GSAI-ML/LLaDA-8B-Base",
    ):
        try:
            from .llada import LLaDAModelLM as LLaDA
            return LLaDA.from_pretrained(model_name, **kwargs)
        except:
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_name, **kwargs)
    else:
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_name, **kwargs)

def get_special_tokens(model_name: str, tokenizer: AutoTokenizer):
    if model_name in (
        "Dream-org/Dream-v0-Instruct-7B",
        "Dream-org/Dream-v0-Base-7B",
        "Gen-Verse/TraDo-4B-Instruct",
        "Gen-Verse/TraDo-8B-Instruct",
        "Gen-Verse/TraDo-8B-Thinking"
    ):
        return {
            'eos_token': 
            {
                'id': tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                'token': tokenizer.special_tokens_map['eos_token']
            },
            'pad_token': {
                'id': tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
                'token': tokenizer.special_tokens_map['pad_token']
            },
            'mask_token': {
                'id': tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                'token': tokenizer.special_tokens_map['mask_token']
            }
        }
    elif model_name in (
        "GSAI-ML/LLaDA-8B-Instruct",
        "GSAI-ML/LLaDA-8B-Base",
    ):
        return {
            'eos_token': {
                'id': tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                'token': "<|endoftext|>"
            },
            'pad_token': {
                'id': tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                'token': "<|endoftext|>"
            },
            'mask_token': {
                'id': tokenizer.convert_tokens_to_ids("<|mdm_mask|>"),
                'token': "<|mdm_mask|>"
            }
        }
    else:
        return {
            'eos_token': {
                'id': tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                'token': tokenizer.special_tokens_map['eos_token']
            },
            'pad_token': {
                'id': tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
                'token': tokenizer.special_tokens_map['pad_token']
            },
            'mask_token': {
                'id': tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                'token': tokenizer.special_tokens_map['mask_token']
            }
        }