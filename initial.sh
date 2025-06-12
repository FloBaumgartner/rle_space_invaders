#!/bin/sh
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16  # num has to be = num_env (not more than 16)
#SBATCH --gres=gpu:0  # no GPU for this run
#SBATCH --partition=performance
#SBATCH --output=out/initial-%A_%a.out
#SBATCH --error=out/initial-%A_%a.err

.venv/bin/python ppo_clean_rl.py --wandb_tracking --exp_name initial