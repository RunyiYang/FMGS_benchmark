#!/bin/bash
#SBATCH --job-name=fmgs
#SBATCH --nodelist=dgx3
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:1         # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --mem=120G
#SBATCH --cpus-per-task=64      # Number of CPU cores per task
#SBATCH --time=1-0         # Job timeout
#SBATCH --output=output_logs/fmgs.log     # Redirect stdout to a log file
#SBATCH --error=output_logs/fmgs.err    # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email

# initialize conda
conda init --all
source ~/.bashrc
source /scratch/runyi_yang/miniconda3/bin/activate
conda activate fmgs

# GPU/CUDA vars
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export TCNN_CUDA_ARCHITECTURES=90
export LD_LIBRARY_PATH=/scratch/runyi_yang/miniconda3/envs/fmgs/lib:$LD_LIBRARY_PATH

# dynamically gather scene IDs
DATA_DIR="/home/runyi_yang/benchmark2025/dataset/Scannetpp_val/original_data"
SCENES=()
for d in "$DATA_DIR"/*; do
  if [ -d "$d" ]; then
    SCENES+=("$(basename "$d")")
  fi
done

echo "Processing ${#SCENES[@]} scenes…"


# Base path for ScanNet scans
SRC_BASE="/home/runyi_yang/benchmark2025/dataset/Scannetpp_val/original_data"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="${SRC_BASE}/${SCENE}"
  MODEL_DIR="${SCENE_DIR}/gsplat"
  mkdir -p "$MODEL_DIR"

  echo "=== Processing ${SCENE} ==="

  # 1) initial checkpoint-only run (was your first scannetpp call)
  # python train.py \
  #   -s "$SCENE_DIR" \
  #   --dataformat scannetpp \
  #   --model_path "$MODEL_DIR" \
  #   --checkpoint_iterations 7000 30000 \
  #   --port 6034

  # # 2) full fine-tune / render-feature run (was your second scannetpp call)
  python train.py \
    -s "$SCENE_DIR" \
    --dataformat scannetpp \
    --model_path "$MODEL_DIR" \
    --opt_vlrenderfeat_from 30000 \
    --test_iterations 32000 32500 \
    --save_iterations 32000 32500 \
    --iterations 32500 \
    --checkpoint_iterations 32000 32500 \
    --start_checkpoint "${MODEL_DIR}/chkpnt30000.pth" \
    --fmap_resolution 2 \
    --lambda_clip 0.2 \
    --fmap_lr 0.005 \
    --fmap_render_radiithre 2 \
    --port 6034

  # python get_3d_features.py -s "$SCENE_DIR" --model_path "$MODEL_DIR" --dataformat holicity --runon_train
done
