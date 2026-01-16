#!/bin/bash
#SBATCH --job-name=rewact_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

# Exit on any error
set -e

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u5dm/pravsels.u5dm"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/rewact"
data_dir="${scratch_dir}/rewact"
container="${data_dir}/container/rewact_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"

# Training config
CONFIG_FILE="configs/train_sam3.yaml"

# Extract job_name from config file
JOB_NAME=$(grep "job_name:" "${CONFIG_FILE}" | awk '{print $NF}' | tr -d '"'\'' ')
LAST_CHECKPOINT="${data_dir}/outputs/train/${JOB_NAME}/checkpoints/last"

mkdir -p "${HF_CACHE}" "${data_dir}/outputs"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

# Auto-resume logic: check if a "last" checkpoint exists on scratch
if [ -d "${LAST_CHECKPOINT}" ]; then
    echo "Found existing checkpoint at ${LAST_CHECKPOINT}. Resuming training..."
    TRAIN_CMD="python -u scripts/train.py \
        --config=${CONFIG_FILE} \
        --output_dir=${data_dir}/outputs/train/${JOB_NAME} \
        --resume=true"
else
    echo "No checkpoint found. Starting fresh training..."
    TRAIN_CMD="python -u scripts/train.py \
        --config=${CONFIG_FILE} \
        --output_dir=${data_dir}/outputs/train/${JOB_NAME} \
        --policy.sam3.weights=${data_dir}/weights/sam3.pt"
fi

# Package Overlay Path (on scratch)
PYTHON_EXT_DIR="${data_dir}/python_packages"

# Ensure it's passed to the container
EXPORT_VARS="export PYTHONPATH=${PYTHON_EXT_DIR}:${repo_dir}:\$PYTHONPATH"
EXPORT_VARS="${EXPORT_VARS} && export LD_LIBRARY_PATH=${PYTHON_EXT_DIR}/torch/lib:\$LD_LIBRARY_PATH"
EXPORT_VARS="${EXPORT_VARS} && export PYTHONUNBUFFERED=1"

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

# Run and capture exit code
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "${EXPORT_VARS} && ${TRAIN_CMD}"

# Capture exit code
EXIT_CODE=$?

end_time="$(date -Is --utc)"

echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

# If training failed, print helpful info
if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    echo "Last checkpoint location: ${LAST_CHECKPOINT}"
    exit ${EXIT_CODE}
fi

exit 0