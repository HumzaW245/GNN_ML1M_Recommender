#!/bin/bash
#SBATCH --job-name=RunsGNN
#SBATCH --output=jobGNN_output.txt
#SBATCH --error=jobGNN_error.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=100Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /home/humza245/projects/def-eugenium/humza245/GNN_ML1M_Recommender/torchDFRenv/bin/activate
export WANDB_MODE=online

#Go into wandb folder and run 'wandb sync /' to sync offline logs to online repo

python eval.py \
--wandbNameSuffix "Test1" --epochs 1 --lr 0.01 --hidden_channels 16 \
--accuracyTheshold 0.5

