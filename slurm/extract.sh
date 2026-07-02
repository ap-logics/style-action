#!/bin/bash
#SBATCH --job-name=sac_extract
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err

# array index 2 = clip control (no motion model needed, but GPU still useful for CLIP)
MODELS=(mdm t2mgpt clip)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Extracting latents for model: $MODEL"

source ~/.bashrc
conda activate sac-eval

cd "$(dirname "$0")/.."
python pipeline.py --model "$MODEL" --stage extract
