#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=test_youcook
#SBATCH --output=log.out
#SBATCH --error=log.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu01
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

export PYTHONWARNINGS="ignore"
export TRANSFORMERS_CACHE=backbone

cd ../vid2seq

python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env vc.py \
    --epochs=1 \
    --lr=3e-4 \
    --save_dir=youcook_vcggt \
    --combine_datasets youcook \
    --combine_datasets_val youcook \
    --batch_size=32 \
    --batch_size_val=1 \
    --schedule="cosine_with_warmup" \
    --max_input_tokens=256 \
    --max_output_tokens=32