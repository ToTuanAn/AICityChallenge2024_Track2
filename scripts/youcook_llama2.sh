#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=test_llama2
#SBATCH --output=log.out
#SBATCH --error=log.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu01
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

export PYTHONWARNINGS="ignore"
export TRANSFORMERS_CACHE=backbone

cd ../vid2seq

python -m torch.distributed.launch --nproc_per_node 8 --use_env vc.py --model_name=<MODEL_DIR>/7BHF \
--save_dir=chapters_vcggt_zeroshotllama --combine_datasets chapters --combine_datasets_val chapters \
--batch_size_val=1 --max_input_tokens=256 --max_output_tokens=32 --eval

