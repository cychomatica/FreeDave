DETERMINISTIC=False
SEED=42

for DATASET in "HumanEval";
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
    for MODEL in "Gen-Verse/TraDo-4B-Instruct"; 
    do
        for FAST_SAMPLING_METHOD in "Parallel" "FreeDave";
        do
            echo "Running $MODEL on $DATASET"
            echo "Fast sampling method: $FAST_SAMPLING_METHOD"
            if [[ "$MODEL" == "Gen-Verse/TraDo-8B-Thinking" ]];
            then
                MAX_TOKEN=30000
            else
                MAX_TOKEN=2048
            fi

            # OmegaConf CLI treats bare None as the string "None"; use lowercase null for real absent values.
            K=null
            TEMPERATURE=0.0
            CONFIDENCE_THRESHOLD=null
            DRAFT_STEPS=1
            DRAFT_MODE=null
            EAGER_ACCEPTANCE_MODE=False

            if [[ "$FAST_SAMPLING_METHOD" == "Parallel" ]];
            then
                TEMPERATURE=1.0
                CONFIDENCE_THRESHOLD=0.9
                K=0
            elif [[ "$FAST_SAMPLING_METHOD" == "FreeDave" ]];
            then
                DRAFT_STEPS=8
                DRAFT_MODE="tree_attention"
                EAGER_ACCEPTANCE_MODE=True
            fi

            for i in {1};
            do
                python -m eval.trado_eval_new \
                config=configs/trado_eval_new.yaml \
                dataset.eval_dataset=$DATASET \
                model=$MODEL \
                experiment.deterministic=$DETERMINISTIC \
                experiment.seed=$SEED \
                rollout.top_k=$K \
                rollout.max_token=$MAX_TOKEN \
                rollout.block_size=4 \
                rollout.draft_steps=$DRAFT_STEPS \
                rollout.draft_mode=$DRAFT_MODE \
                rollout.confidence_threshold=$CONFIDENCE_THRESHOLD \
                rollout.eager_acceptance_mode=$EAGER_ACCEPTANCE_MODE \
                rollout.temperature=$TEMPERATURE \
                dataset.data_type=$DATA_TYPE
            done
        done
    done
done