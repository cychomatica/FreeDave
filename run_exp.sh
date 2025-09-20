DATASET="MATH500"
MODEL="Gen-Verse/TraDo-4B-Instruct"

CUDA_VISIBLE_DEVICES=0 python trado_eval.py \
config=configs/trado_eval.yaml \
dataset.eval_dataset=$DATASET \
model=$MODEL \
rollout.max_token=2048 \
rollout.block_size=4 \
rollout.denoising_steps_per_block=4 \
rollout.draft_steps=1

CUDA_VISIBLE_DEVICES=0 python trado_eval.py \
config=configs/trado_eval.yaml \
dataset.eval_dataset=$DATASET \
model=$MODEL \
rollout.max_token=2048 \
rollout.block_size=4 \
rollout.denoising_steps_per_block=4 \
rollout.draft_steps=4 \
rollout.fast_sampling_version=v0

CUDA_VISIBLE_DEVICES=0 python trado_eval.py \
config=configs/trado_eval.yaml \
dataset.eval_dataset=$DATASET \
model=$MODEL \
rollout.max_token=2048 \
rollout.block_size=4 \
rollout.denoising_steps_per_block=4 \
rollout.draft_steps=8 \
rollout.fast_sampling_version=v1
