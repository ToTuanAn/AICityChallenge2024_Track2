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

cd ../vid2seq/backbone

git clone git@hf.co:meta-llama/Llama-2-7b-hf

cd Llama-2-7b-hf
rm -rf .git