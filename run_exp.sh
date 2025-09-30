#! /bin/bash

# DATASETS=("MATH500" "GSM8K" "AIME2024")
# MODELS=("Gen-Verse/TraDo-4B-Instruct" "Gen-Verse/TraDo-8B-Instruct" "Gen-Verse/TraDo-8B-Thinking" "JetLM/SDAR-4B-Chat" "JetLM/SDAR-8B-Chat" "Dream-org/Dream-v0-Instruct-7B")
# FAST_SAMPLING_METHODS=("NA" "Dynamic" "FreeDave")

MODEL="Gen-Verse/TraDo-4B-Instruct"
DATASET="MATH500"
FAST_SAMPLING_METHOD="FreeDave"

echo "Running $MODEL on $DATASET"
echo "Fast sampling method: $FAST_SAMPLING_METHOD"

if [[ "$MODEL" == "Gen-Verse/TraDo-4B-Instruct" || "$MODEL" == "Gen-Verse/TraDo-8B-Instruct" || "$MODEL" == "Gen-Verse/TraDo-8B-Thinking" ]]; 
then

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
    EAGER_ACCEPTANCE_MODE=False

    if [[ "$FAST_SAMPLING_METHOD" == "Dynamic" ]];
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
        EAGER_ACCEPTANCE_MODE=True
    fi

    python -m eval.trado_eval \
    config=configs/trado_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.max_token=$MAX_TOKEN \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=$DENOISING_STEPS_PER_BLOCK \
    rollout.draft_steps=$DRAFT_STEPS \
    rollout.top_k=$K \
    rollout.remasking_strategy=$REMASKING_STRATEGY \
    rollout.fast_sampling_version=$FAST_SAMPLING_VERSION \
    rollout.eager_acceptance_mode=$EAGER_ACCEPTANCE_MODE

elif [[ "$MODEL" == "JetLM/SDAR-4B-Chat" || "$MODEL" == "JetLM/SDAR-8B-Chat" ]]; 
then

    MAX_TOKEN=2048
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

    python -m eval.trado_eval \
    config=configs/sdar_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.max_token=$MAX_TOKEN \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=$DENOISING_STEPS_PER_BLOCK \
    rollout.draft_steps=$DRAFT_STEPS \
    rollout.top_k=$K \
    rollout.remasking_strategy=$REMASKING_STRATEGY \
    rollout.fast_sampling_version=$FAST_SAMPLING_VERSION

elif [[ "$MODEL" == "Dream-org/Dream-v0-Instruct-7B" ]]; 
then

    MAX_TOKEN=1600
    STEPS=1600
    K=1
    REMASKING_STRATEGY="low_confidence_static"
    DRAFT_STEPS=1
    BLOCK_SIZE=32

    if [[ "$FAST_SAMPLING_METHOD" == "Naive" ]];
    then
        STEPS=800
    elif [[ "$FAST_SAMPLING_METHOD" == "Dynamic" ]];
    then
        K=0
        REMASKING_STRATEGY="low_confidence_dynamic"
        BLOCK_SIZE=4
    elif [[ "$FAST_SAMPLING_METHOD" == "FreeDave" ]];
    then
        DRAFT_STEPS=4
    fi    

    python -m eval.dream_eval \
    config=configs/dream_eval.yaml \
    dataset.eval_dataset=$DATASET \
    model=$MODEL \
    rollout.steps=$STEPS \
    rollout.max_gen_length=$MAX_TOKEN \
    rollout.block_size=$BLOCK_SIZE \
    rollout.draft_steps=$DRAFT_STEPS \
    rollout.top_k=$K \
    rollout.remasking_strategy=$REMASKING_STRATEGY

else
    echo "Error: Model $MODEL not supported yet."
    exit 1
fi

echo "Evaluation for $MODEL on $DATASET completed."