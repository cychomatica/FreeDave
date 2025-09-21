DATASET="MATH500"
# DATASET="GSM8K"
# DATASET="AIME2024"
MODEL="Gen-Verse/TraDo-4B-Instruct"
# MODEL="JetLM/SDAR-4B-Chat"
# MODEL="Dream-org/Dream-v0-Instruct-7B"
# MODEL="GSAI-ML/LLaDA-8B-Instruct"

if [[ "$MODEL" == "Gen-Verse/TraDo-4B-Instruct" || "$MODEL" == "Gen-Verse/TraDo-8B-Instruct" || "$MODEL" == "Gen-Verse/TraDo-8B-Thinking" ]]; then
    CUDA_VISIBLE_DEVICES=3 python trado_eval.py \
    config=configs/trado_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.max_token=2048 \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=4 \
    rollout.draft_steps=4 \
    rollout.fast_sampling_version=v0
fi

elif [[ "$MODEL" == "JetLM/SDAR-4B-Chat" || "$MODEL" == "JetLM/SDAR-8B-Chat"]]; then
    CUDA_VISIBLE_DEVICES=3 python trado_eval.py \
    config=configs/sdar_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.max_token=2048 \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=4 \
    rollout.draft_steps=4 \
    rollout.fast_sampling_version=v0
fi

elif [[ "$MODEL" == "Dream-org/Dream-v0-Instruct-7B" ]]; then
    CUDA_VISIBLE_DEVICES=3 python dream_eval.py \
    config=configs/dream_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.steps=1024 \
    rollout.max_gen_length=1024 \
    rollout.block_size=32 \
    rollout.draft_steps=4
fi

elif [[ "$MODEL" == "GSAI-ML/LLaDA-8B-Instruct" ]]; then
    CUDA_VISIBLE_DEVICES=3 python llada_eval.py \
    config=configs/llada_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.steps=1024 \
    rollout.max_gen_length=1024 \
    rollout.block_size=32 \
    rollout.draft_steps=4
fi


# CUDA_VISIBLE_DEVICES=3 python trado_eval.py \
# config=configs/trado_eval.yaml \
# dataset.eval_dataset=$DATASET \
# model=$MODEL \
# rollout.max_token=2048 \
# rollout.block_size=4 \
# rollout.denoising_steps_per_block=4 \
# rollout.draft_steps=4 \
# rollout.fast_sampling_version=v0

# CUDA_VISIBLE_DEVICES=0 python trado_eval.py \
# config=configs/trado_eval.yaml \
# dataset.eval_dataset=$DATASET \
# model=$MODEL \
# rollout.max_token=2048 \
# rollout.block_size=4 \
# rollout.denoising_steps_per_block=4 \
# rollout.draft_steps=4 \
# rollout.fast_sampling_version=v0

# CUDA_VISIBLE_DEVICES=0 python trado_eval.py \
# config=configs/trado_eval.yaml \
# dataset.eval_dataset=$DATASET \
# model=$MODEL \
# rollout.max_token=2048 \
# rollout.block_size=4 \
# rollout.denoising_steps_per_block=4 \
# rollout.draft_steps=8 \
# rollout.fast_sampling_version=v1

# CUDA_VISIBLE_DEVICES=0 python dream_eval.py \
# config=configs/dream_eval.yaml \
# dataset.eval_dataset=$DATASET \
# dataset.data_type="math" \
# model=$MODEL \
# rollout.draft_steps=1

# CUDA_VISIBLE_DEVICES=1 python dream_eval.py \
# config=configs/dream_eval.yaml \
# dataset.eval_dataset=$DATASET \
# dataset.data_type="math" \
# model=$MODEL \
# rollout.draft_steps=2 \
# rollout.fast_sampling_version=v0