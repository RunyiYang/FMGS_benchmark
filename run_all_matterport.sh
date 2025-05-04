# # Initialize conda and activate your env
source ~/.bashrc
conda init --all
conda activate fmgs

# CUDA & library setup
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export TCNN_CUDA_ARCHITECTURES=90
export LD_LIBRARY_PATH=/scratch/runyi_yang/miniconda3/envs/fmgs/lib:$LD_LIBRARY_PATH

# List of your 12 ScanNet scenes
SCENES=(
  2t7WUuJeko7_02
  5ZKStnWn8Zo_15
  ARNzJeq3xxb_07
  fzynW3qQPVF_00
  jtcxE69GiFV_32
  pa4otMbVnkk_27
  q9vSo1VnCiC_05
  rqfALeAoiTq_11
  UwV83HsGsw3_22
  wc2JMjhGNzB_11
)

# conda init --all
# conda activate fmgs

# Base path for ScanNet scans
SRC_BASE="/home/runyi_yang/benchmark2025/dataset/matterport_val_subset/original_data"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="${SRC_BASE}/${SCENE}"
  MODEL_DIR="${SCENE_DIR}/gsplat"
  mkdir -p "$MODEL_DIR"

  echo "=== Processing ${SCENE} ==="

  # 1) initial checkpoint-only run (was your first scannetpp call)
  python train.py \
    -s "$SCENE_DIR" \
    --dataformat matterport \
    --model_path "$MODEL_DIR" \
    --checkpoint_iterations 7000 30000 \
    --port 6034

  # 2) full fine-tune / render-feature run (was your second scannetpp call)
  python train.py \
    -s "$SCENE_DIR" \
    --dataformat matterport \
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
done
