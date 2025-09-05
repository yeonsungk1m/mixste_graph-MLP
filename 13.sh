#!/bin/bash
#SBATCH --job-name=MIXSTE
#SBATCH -A gpu 
#SBATCH --partition=gpu-4farm
#SBATCH --gres=gpu:4                 # GPU 4장 예약(필수)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24           # 4개 프로세스 × 4코어 = 16
#SBATCH --mem=64G                    # 4실험 동시 실행이면 48~64G 권장
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%A_out.log
#SBATCH --error=logs/%A_err.log

set -euo pipefail
mkdir -p logs

# ---- 실험 큐: 여기만 편집하세요 ----
WINDOWS=(1 5 7 10 15 20 25 30)  # 예: 총 16개(= 34h × 4웨이브)
MAX_PARALLEL=4
PER_STEP_CPUS=6
LOGDIR_PREFIX="log/run"    # 파이썬 스크립트 내부 로그 디렉토리
CKPT_PREFIX="checkpoint"   # 체크포인트 기본 경로
# ------------------------------------

# 동시 실행을 4개로 유지하는 간단한 쓰로틀 함수
throttle() {
  # bash 5 이상이면 wait -n 사용이 가장 깔끔
  if wait -n 2>/dev/null; then
    return 0
  fi
  # 구형 bash 호환: 백그라운드 작업 수가 MAX_PARALLEL 미만이 될 때까지 대기
  while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 30
  done
}

# 큐 실행
idx=0
for W in "${WINDOWS[@]}"; do
  idx=$((idx+1))
  # 동시 실행 상한 도달 시, 하나 끝날 때까지 대기
  while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
    throttle
  done

  # 각 실험마다 고유 로그/디렉토리로 분리(충돌 방지)
  RUN_TAG="w${W}_#${idx}"
  PY_LOG="${LOGDIR_PREFIX}_${RUN_TAG}"
  OUT_LOG="logs/${SLURM_JOB_ID}_${RUN_TAG}.out"

  echo "[LAUNCH] ${RUN_TAG} at $(date)"

  srun --exclusive -N1 -n1 --gres=gpu:1 -c "${PER_STEP_CPUS}" \
    python run-seal.py \
      -k gt -f 243 -s 243 \
      -l "${PY_LOG}" \
      -c "${CKPT_PREFIX}/243-cpn-gt-seal13_${W}_1e3" \
      --lr_loss 1e-4 \
      --energy_weight 1e-5 \
      --em_loss_type margin --margin_type mpjpe \
      --energy_pair_weight 1e-3 --energy_pair_kappa 1.0 \
      --energy_pair_window "${W}" \
    > "${OUT_LOG}" 2>&1 &
done

# 남은 작업 마무리 대기
wait
echo "[ALL DONE] $(date)"