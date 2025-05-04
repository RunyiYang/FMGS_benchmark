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
  ytwUEEljP6RgoV0MviqvsQ_LD
  yxYrxgMjOWCeJ3a_KyBXKA_LD
  z0IhWSN8uhWDltDqmwdY8g_HD
  z5af-sE80KZQc_9Se7Aq_g_LD
  z8hAivfNyPaRSobbXCR8jQ_LD
  yuVpITQv74rpaWqX4kEW4Q_HD
  yy03mA02i0OxVdMb9klRgg_LD
  z37U-BzqOo0_mSeJh7gXbg_LD
  z5l-XERj7c-N4Rdk4JSLeg_LD
  z9UseFv2rAwNfe21MuYxZQ_LD
)

# conda init --all
# conda activate fmgs

# Base path for ScanNet scans
SRC_BASE="/home/runyi_yang/benchmark2025/dataset/holicity_val_subset_mini/original_data"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="${SRC_BASE}/${SCENE}"
  MODEL_DIR="${SCENE_DIR}/gsplat"
  mkdir -p "$MODEL_DIR"

  echo "=== Processing ${SCENE} ==="

  # 1) initial checkpoint-only run (was your first scannetpp call)
  # python train.py \
  #   -s "$SCENE_DIR" \
  #   --dataformat holicity \
  #   --model_path "$MODEL_DIR" \
  #   --checkpoint_iterations 7000 30000 \
  #   --port 6034

  # # 2) full fine-tune / render-feature run (was your second scannetpp call)
  # python train.py \
  #   -s "$SCENE_DIR" \
  #   --dataformat holicity \
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

  python get_3d_features.py -s "$SCENE_DIR" --model_path "$MODEL_DIR" --dataformat holicity --runon_train
done
