# The modeling files on huggingface do not directly support flex attention, so we wrap them locally.
# For each DLM, we basically copy the original modeling file and make some tweaks to support flex attention.

def get_model(model_name: str, **kwargs):
    if model_name == "Dream-org/Dream-v0-Instruct-7B":
        from .dream import DreamModel as Dream
        return Dream.from_pretrained(model_name, **kwargs)
    elif model_name == "Gen-Verse/TraDo-4B-Instruct" or "Gen-Verse/TraDo-8B-Instruct":
        from .trado import SDARForCausalLM as TraDo
        return TraDo.from_pretrained(model_name, **kwargs)
    elif model_name == "GSAI-ML/LLaDA-8B-Instruct":
        from .llada import LLaDAModelLM as LLaDA
        return LLaDA.from_pretrained(model_name, **kwargs)
    else:
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_name, **kwargs)