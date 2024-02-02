#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=baseline_pedestrian
#SBATCH --output=logs/baseline_pedestrian.out
#SBATCH --error=logs/baseline_pedestrian.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu01
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

export PYTHONWARNINGS="ignore"
export TRANSFORMERS_CACHE=backbone

cd ../vid2seq

python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    vc.py \
    --epochs=2000 \
    --lr=3e-4 \
    --save_dir=baseline_pedestrian \
    --combine_datasets wts \
    --combine_datasets_val wts \
    --no_speech \
    --batch_size=16 \
    --batch_size_val=1 \
    --eval_skip 100 \
    --schedule="cosine_with_warmup" \
    --max_input_tokens=256 \
    --max_output_tokens=256 \
    --wandb_project vid2seq \
    --wandb_name baseline_pedestrian \