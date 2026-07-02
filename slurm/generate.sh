#!/bin/bash
#SBATCH --job-name=sac_generate
#SBATCH --array=0-63
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=logs/generate_%A_%a.out
#SBATCH --error=logs/generate_%A_%a.err

# Pass --export=MODEL=mdm or MODEL=t2mgpt when submitting:
#   sbatch --export=MODEL=mdm slurm/generate.sh
MODEL=${MODEL:-mdm}

echo "Generating motions for model=$MODEL prompt_idx=$SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate sac-eval

cd "$(dirname "$0")/.."
python pipeline.py \
    --model "$MODEL" \
    --stage generate \
    --prompt_idx "$SLURM_ARRAY_TASK_ID" \
    --n_seeds 5
