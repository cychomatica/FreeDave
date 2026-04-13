#!/bin/bash
model=Gen-Verse/TraDo-4B-Instruct
model_basename=$(basename $model)

tasks="humaneval_instruct mbpp_instruct minerva_math500 gsm8k"
nshots="0 3 4 5"
lengths="2048 2048 2048 2048"
block_lengths="4 4 4 4"
temperatures="0 0 0 0" # sampling temperature; default 0
alg_temps="0 0 0 0" # unmasking temperature; -1: l2r, 0: static
confidence_thresholds="0.9 0.9 0.9 0.9"
draft_steps="8 8 8 8"

# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra BLOCK_LENGTH_ARRAY <<< "$block_lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra ALG_TEMP_ARRAY <<< "$alg_temps"
read -ra CONFIDENCE_THRESHOLDS_ARRAY <<< "$confidence_thresholds"
read -ra DRAFT_STEPS_ARRAY <<< "$draft_steps"
export HF_ALLOW_CODE_EVAL=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

EARLY_EXIT=True
DRAFT_MODE="tree_attention" # "tree_attention" or "batch_expanding"
EAGER_ACCEPTANCE_MODE=True
DETERMINISTIC=True

# Iterate through the tasks array
for i in "${!TASKS_ARRAY[@]}"; do
    # base static decoding
    output_path=evals_results/${model_basename}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-base
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Model: ${model_basename}, Decoding Strategy: base, alg_temp: ${ALG_TEMP_ARRAY[$i]}"
    echo "Output: $output_path"
    accelerate launch eval_trado.py --model trado \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},block_length=${BLOCK_LENGTH_ARRAY[$i]},temperature=${TEMP_ARRAY[$i]},alg_temp=${ALG_TEMP_ARRAY[$i]},early_exit=${EARLY_EXIT},deterministic=${DETERMINISTIC} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template
    # confidence-aware parallel decoding
    output_path=evals_results/${model_basename}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-parallel-tau=${CONFIDENCE_THRESHOLDS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Model: ${model_basename}, Decoding Strategy: parallel-tau=${CONFIDENCE_THRESHOLDS_ARRAY[$i]}, alg_temp: ${ALG_TEMP_ARRAY[$i]}"
    echo "Output: $output_path"
    accelerate launch eval_trado.py --model trado \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},block_length=${BLOCK_LENGTH_ARRAY[$i]},temperature=${TEMP_ARRAY[$i]},alg_temp=${ALG_TEMP_ARRAY[$i]},confidence_threshold=${CONFIDENCE_THRESHOLDS_ARRAY[$i]},early_exit=${EARLY_EXIT},deterministic=${DETERMINISTIC} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template
    # FreeDave decoding
    output_path=evals_results/${model_basename}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-freedave-d=${DRAFT_STEPS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Model: ${model_basename}, Decoding Strategy: freedave-d=${DRAFT_STEPS_ARRAY[$i]}, alg_temp: ${ALG_TEMP_ARRAY[$i]}"
    echo "Output: $output_path"
    accelerate launch eval_trado.py --model trado \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},block_length=${BLOCK_LENGTH_ARRAY[$i]},temperature=${TEMP_ARRAY[$i]},alg_temp=${ALG_TEMP_ARRAY[$i]},decoding_alg=freedave,draft_steps=${DRAFT_STEPS_ARRAY[$i]},draft_mode=${DRAFT_MODE},eager_acceptance_mode=${EAGER_ACCEPTANCE_MODE},early_exit=${EARLY_EXIT},deterministic=${DETERMINISTIC} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template
done