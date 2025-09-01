#!/bin/bash
#SBATCH --job-name=MIXSTE
#SBATCH --partition=gpu-1farm        # 보장 파티션
#SBATCH --gres=gpu:1             # H100 4장
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # torchrun에서 nproc_per_node로 제어
#SBATCH --cpus-per-task=4             # DataLoader용 스레드 권장치
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%A_out.log
#SBATCH --error=logs/%A_error.log

python run.py -k gt -f 243 -s 243 -l log/run -c checkpoint/243-gt -gpu 0
