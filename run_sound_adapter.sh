#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=32g
#SBATCH -J "ESC_10_sound_adapter"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbiswas@wpi.edu
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100


source venv/bin/activate
# python -W ignore extract_feature.py
python -W ignore train_sound_adapter.py
python -W ignore test_sound_adapter.py
deactivate