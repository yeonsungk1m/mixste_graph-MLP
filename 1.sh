#!/bin/bash
#SBATCH --job-name=graphformer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=P2
#SBATCH --output=./logs/%A_out.log
#SBATCH --error=./logs/%A_error.log

python run.py -k gt -f 243 -s 243 -l log/run -c checkpoint/243-gt -gpu 0