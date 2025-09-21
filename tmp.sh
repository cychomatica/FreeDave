DATASETS=("MATH500" "GSM8K" "AIME2024")
MODELS=("Gen-Verse/TraDo-4B-Instruct" "Gen-Verse/TraDo-8B-Instruct" "Gen-Verse/TraDo-8B-Thinking" "JetLM/SDAR-4B-Chat" "JetLM/SDAR-8B-Chat" "Dream-org/Dream-v0-Instruct-7B" "GSAI-ML/LLaDA-8B-Instruct")
FAST_SAMPLING_METHODS=("NA" "Naive" "Dynamic" "FreeDave" "FreeDave++")

MODEL=${MODELS[0]}
DATASET=${DATASETS[0]}
FAST_SAMPLING_METHOD=${FAST_SAMPLING_METHODS[0]}

echo "Running $MODEL on $DATASET with $FAST_SAMPLING_METHOD"