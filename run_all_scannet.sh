# # Initialize conda and activate your env
# source ~/.bashrc
# conda init --all
# conda activate fmgs

# CUDA & library setup
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export TCNN_CUDA_ARCHITECTURES=90
export LD_LIBRARY_PATH=/scratch/runyi_yang/miniconda3/envs/fmgs/lib:$LD_LIBRARY_PATH

# List of your 12 ScanNet scenes
SCENES=(
  scene0011_00
  scene0011_01
  scene0153_00
  scene0153_01
  scene0329_01
  scene0329_02
  scene0435_00
  scene0435_01
  scene0578_01
  scene0578_02
  scene0651_01
  scene0651_02
)

# conda init --all
# conda activate fmgs

# Base path for ScanNet scans
SRC_BASE="/home/runyi_yang/benchmark2025/dataset/ScanNet/scans"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="${SRC_BASE}/${SCENE}"
  MODEL_DIR="${SCENE_DIR}/gsplat"
  mkdir -p "$MODEL_DIR"

  echo "=== Processing ${SCENE} ==="

  # # 1) initial checkpoint-only run (was your first scannetpp call)
  # python train.py \
  #   -s "$SCENE_DIR" \
  #   --dataformat scannet \
  #   --model_path "$MODEL_DIR" \
  #   --checkpoint_iterations 7000 30000 \
  #   --port 6034

  # # 2) full fine-tune / render-feature run (was your second scannetpp call)
  # python train.py \
  #   -s "$SCENE_DIR" \
  #   --dataformat scannet \
  #   --model_path "$MODEL_DIR" \
  #   --opt_vlrenderfeat_from 30000 \
  #   --test_iterations 32000 32500 \
  #   --save_iterations 32000 32500 \
  #   --iterations 32500 \
  #   --checkpoint_iterations 32000 32500 \
  #   --start_checkpoint "${MODEL_DIR}/chkpnt30000.pth" \
  #   --fmap_resolution 2 \
  #   --lambda_clip 0.2 \
  #   --fmap_lr 0.005 \
  #   --fmap_render_radiithre 2 \
  #   --port 6034
  python get_3d_features.py -s "$SCENE_DIR" --model_path "$MODEL_DIR" --dataformat scannet --runon_train
done
