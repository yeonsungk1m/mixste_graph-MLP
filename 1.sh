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

python run-seal.py -k gt -f 243 -s 243 -l log/run -c checkpoint/243-cpn-gt-seal13_3_1e3 -gpu 0 \
	--lr_loss 1e-4 --energy_weight 1e-5 --em_loss_type margin --margin_type mpjpe --energy_pair_weight 1e-3 --energy_pair_kappa 1.0 --energy_pair_window 3


python run.py -k gt -f 3 -s 3 -l log/run -c checkpoint/3-base -gpu 0

python run-seal.py -k gt -f 3 -s 3 -l log/run -c checkpoint/3-MLP -gpu 0 \
--lr_loss 1e-4 --energy_weight 1e-5 --em_loss_type margin --margin_type mpjpe --energy_pair_weight 1e-3 --energy_pair_kappa 1.0 --energy_pair_window 3

python run-seal.py -k gt -f 3 -s 3 -l log/run -c checkpoint/3-GRAPH -gpu 0 \
--lr_loss 1e-4 --energy_weight 1e-5 --em_loss_type margin --margin_type mpjpe --energy_pair_weight 1e-3 --energy_pair_kappa 1.0 --energy_pair_window 3
