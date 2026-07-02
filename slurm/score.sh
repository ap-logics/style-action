#!/bin/bash
#SBATCH --job-name=sac_score
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err

# Depends on both extract and generate completing first.
# Submit as:
#   sbatch --dependency=afterok:<extract_job_id>,<generate_job_id> slurm/score.sh

MODEL=${MODEL:-mdm}

echo "Scoring model=$MODEL"

source ~/.bashrc
conda activate sac-eval

cd "$(dirname "$0")/.."
python pipeline.py --model "$MODEL" --stage score --n_perms 1000
