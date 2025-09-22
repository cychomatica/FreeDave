DATASETS=("MATH500" "GSM8K" "AIME2024")
MODELS=("Gen-Verse/TraDo-4B-Instruct" "Gen-Verse/TraDo-8B-Instruct" "Gen-Verse/TraDo-8B-Thinking" "JetLM/SDAR-4B-Chat" "JetLM/SDAR-8B-Chat" "Dream-org/Dream-v0-Instruct-7B" "GSAI-ML/LLaDA-8B-Instruct")
FAST_SAMPLING_METHODS=("NA" "Naive" "Dynamic" "FreeDave" "FreeDave++")

MODEL="Gen-Verse/TraDo-4B-Instruct"
DATASET="MATH500"
FAST_SAMPLING_METHOD="FreeDave++"

if [[ "$MODEL" == "Gen-Verse/TraDo-4B-Instruct" || "$MODEL" == "Gen-Verse/TraDo-8B-Instruct" || "$MODEL" == "Gen-Verse/TraDo-8B-Thinking" ]]; 
then
    echo "Running $MODEL on $DATASET"
    echo "Fast sampling method: $FAST_SAMPLING_METHOD"
    if [[ "$MODEL" == "Gen-Verse/TraDo-8B-Thinking" ]];
    then
        MAX_TOKEN=30000
    else
        MAX_TOKEN=2048
    fi

    REMASKING_STRATEGY="low_confidence_static"
    FAST_SAMPLING_VERSION="NA"
    K=1
    DRAFT_STEPS=1
    DENOISING_STEPS_PER_BLOCK=4

    if [[ "$FAST_SAMPLING_METHOD" == "Naive" ]];
    then
        DENOISING_STEPS_PER_BLOCK=2
    elif [[ "$FAST_SAMPLING_METHOD" == "Dynamic" ]];
    then
        REMASKING_STRATEGY="low_confidence_dynamic"
        K=0
    elif [[ "$FAST_SAMPLING_METHOD" == "FreeDave" ]];
    then
        DRAFT_STEPS=4
        FAST_SAMPLING_VERSION="v0"
    elif [[ "$FAST_SAMPLING_METHOD" == "FreeDave++" ]];
    then
        DRAFT_STEPS=8
        FAST_SAMPLING_VERSION="v1"
    fi
    CUDA_VISIBLE_DEVICES=2 python trado_eval.py \
    config=configs/trado_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.max_token=$MAX_TOKEN \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=$DENOISING_STEPS_PER_BLOCK \
    rollout.draft_steps=$DRAFT_STEPS \
    rollout.top_k=$K \
    rollout.remasking_strategy=$REMASKING_STRATEGY \
    rollout.fast_sampling_version=$FAST_SAMPLING_VERSION
elif [[ "$MODEL" == "JetLM/SDAR-4B-Chat" || "$MODEL" == "JetLM/SDAR-8B-Chat" ]]; 
then
    echo "Running $MODEL on $DATASET"
    CUDA_VISIBLE_DEVICES=3 python trado_eval.py \
    config=configs/sdar_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.max_token=2048 \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=4 \
    rollout.draft_steps=4 \
    rollout.fast_sampling_version=v0
elif [[ "$MODEL" == "Dream-org/Dream-v0-Instruct-7B" ]]; 
then
    echo "Running $MODEL on $DATASET"
    CUDA_VISIBLE_DEVICES=3 python dream_eval.py \
    config=configs/dream_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.steps=1024 \
    rollout.max_gen_length=1024 \
    rollout.block_size=32 \
    rollout.draft_steps=4
elif [[ "$MODEL" == "GSAI-ML/LLaDA-8B-Instruct" ]]; 
then
    echo "Running $MODEL on $DATASET"
    CUDA_VISIBLE_DEVICES=3 python llada_eval.py \
    config=configs/llada_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.steps=1024 \
    rollout.max_gen_length=1024 \
    rollout.block_size=32 \
    rollout.draft_steps=4
else
    echo "Model $MODEL not found"
fi

echo "Evaluation for $MODEL on $DATASET completed"



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