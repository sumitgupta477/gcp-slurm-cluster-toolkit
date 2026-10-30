#!/bin/bash
#SBATCH --job-name=train-to-vertex
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Train on one L4 GPU, save the model to /home, and upload it to Cloud Storage.
#
# Submit from this folder (the job looks for train.py next to where you ran sbatch):
#   sbatch --export=ALL,GCS_MODEL_URI=gs://<BUCKET>/models/sentiment-v1 train.sh
# Extra arguments go to train.py, e.g.:
#   sbatch --export=ALL,GCS_MODEL_URI=gs://<BUCKET>/models/sentiment-v2 train.sh --epochs 2
set -euo pipefail

: "${GCS_MODEL_URI:?Set GCS_MODEL_URI, e.g. sbatch --export=ALL,GCS_MODEL_URI=gs://BUCKET/models/sentiment-v1 train.sh}"
VENV="${VENV:-$HOME/venv-train}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/models/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"

echo "Job ${SLURM_JOB_ID} on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python "${SLURM_SUBMIT_DIR}/train.py" \
  --output_dir "${OUTPUT_DIR}" \
  --gcs_uri "${GCS_MODEL_URI}" \
  "$@"
