# The modeling files on huggingface do not directly support flex attention, so we wrap them locally.
# For each DLM, we basically copy the original modeling file and make some tweaks to support flex attention.

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