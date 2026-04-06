#!/bin/bash
MODEL="Dream-org/Dream-v0-Instruct-7B"
DETERMINISTIC=False
SEED=42

for DATASET in "MMLU-Pro";
do
    if [[ "$DATASET" == "MMLU-Pro" ]];
    then
    DATA_TYPE="option"
    elif [[ "$DATASET" == "MATH500" || "$DATASET" == "MATH500_subset" || "$DATASET" == "MATH500_miniset" || "$DATASET" == "GSM8K" || "$DATASET" == "AIME2024" || "$DATASET" == "AIME2025" ]];
    then
    DATA_TYPE="math"
    elif [[ "$DATASET" == "MBPP" || "$DATASET" == "HumanEval" || "$DATASET" == "HumanEval_miniset" ]];
    then
    DATA_TYPE="code"
    else
        echo "Error: Dataset $DATASET not supported yet."
        exit 1
    fi
    for BLOCK_SIZE in 32;
    do
        for FAST_SAMPLING_METHOD in "NA" "Parallel" "FreeDave";
        do
            echo "Running $MODEL on $DATASET"
            echo "Fast sampling method: $FAST_SAMPLING_METHOD"
            MAX_TOKEN=1600
            K=null
            TEMPERATURE=0.0
            CONFIDENCE_THRESHOLD=null
            DRAFT_STEPS=1
            DRAFT_MODE=null
            EAGER_ACCEPTANCE_MODE=False
            DUAL_CACHE=True

            if [[ "$FAST_SAMPLING_METHOD" == "Parallel" ]];
            then
                TEMPERATURE=0.1
                K=0
                CONFIDENCE_THRESHOLD=0.95
            elif [[ "$FAST_SAMPLING_METHOD" == "FreeDave" ]];
            then
                DRAFT_STEPS=4
                DRAFT_MODE="tree_attention"
                if [[ "$BLOCK_SIZE" == "4" ]]; then
                    EAGER_ACCEPTANCE_MODE=True
                fi
            fi
            python -m eval.dream_eval_new \
            config=configs/dream_eval_new.yaml \
            dataset.eval_dataset=$DATASET \
            model=$MODEL \
            experiment.deterministic=$DETERMINISTIC \
            experiment.seed=$SEED \
            rollout.top_k=$K \
            rollout.max_token=$MAX_TOKEN \
            rollout.block_size=$BLOCK_SIZE \
            rollout.draft_steps=$DRAFT_STEPS \
            rollout.draft_mode=$DRAFT_MODE \
            rollout.confidence_threshold=$CONFIDENCE_THRESHOLD \
            rollout.eager_acceptance_mode=$EAGER_ACCEPTANCE_MODE \
            rollout.temperature=$TEMPERATURE \
            rollout.dual_cache=$DUAL_CACHE \
            dataset.data_type=$DATA_TYPE
        done
    done
done