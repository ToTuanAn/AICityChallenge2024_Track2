#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=download_model
#SBATCH --output=logs/download_model.out
#SBATCH --error=logs/download_model.out
#SBATCH --gres=gpu:0
#SBATCH --nodelist=gpu01
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

cd ..
python download_model.py \
    --model_id google/flan-t5-large \
    --cache_dir cache \
    --save_dir vid2seq/backbone