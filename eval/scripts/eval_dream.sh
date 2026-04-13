#!/bin/bash
model=Dream-org/Dream-v0-Instruct-7B
model_basename=$(basename $model)

tasks="humaneval_instruct mbpp_instruct minerva_math500 gsm8k"
nshots="0 3 4 5"
lengths="1024 1024 1024 1024"
block_lengths="32 32 32 32"
temperatures="0 0 0 0" # sampling temperature; default 0
alg_temps="0 0 0 0" # unmasking temperature; -1: l2r, 0: static
confidence_thresholds="0.95 0.95 0.95 0.95"
draft_steps="4 4 4 4"

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
DRAFT_MODE="tree_attention"
EAGER_ACCEPTANCE_MODE=False
DUAL_CACHE=True
DETERMINISTIC=True

# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    # base decoding
    output_path=evals_results/${model_basename}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-base
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Model: ${model_basename}, Decoding Strategy: base, alg_temp: ${ALG_TEMP_ARRAY[$i]}"
    echo "Output: $output_path"
    accelerate launch eval_dream.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},block_length=${BLOCK_LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},alg_temp=${ALG_TEMP_ARRAY[$i]},dual_cache=${DUAL_CACHE},early_exit=${EARLY_EXIT},deterministic=${DETERMINISTIC} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
    # confidence-aware parallel decoding
    output_path=evals_results/${model_basename}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-parallel-tau=${CONFIDENCE_THRESHOLDS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Model: ${model_basename}, Decoding Strategy: parallel-tau=${CONFIDENCE_THRESHOLDS_ARRAY[$i]}, alg_temp: ${ALG_TEMP_ARRAY[$i]}"
    echo "Output: $output_path"
    accelerate launch eval_dream.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},block_length=${BLOCK_LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},alg_temp=${ALG_TEMP_ARRAY[$i]},dual_cache=${DUAL_CACHE},confidence_threshold=${CONFIDENCE_THRESHOLDS_ARRAY[$i]},early_exit=${EARLY_EXIT},deterministic=${DETERMINISTIC} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
    # FreeDave decoding
    output_path=evals_results/${model_basename}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-freedave-d=${DRAFT_STEPS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Model: ${model_basename}, Decoding Strategy: freedave-d=${DRAFT_STEPS_ARRAY[$i]}, alg_temp: ${ALG_TEMP_ARRAY[$i]}"
    echo "Output: $output_path"
    accelerate launch eval_dream.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},block_length=${BLOCK_LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},alg_temp=${ALG_TEMP_ARRAY[$i]},dual_cache=${DUAL_CACHE},decoding_alg=freedave,draft_steps=${DRAFT_STEPS_ARRAY[$i]},draft_mode=${DRAFT_MODE},eager_acceptance_mode=${EAGER_ACCEPTANCE_MODE},early_exit=${EARLY_EXIT},deterministic=${DETERMINISTIC} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done