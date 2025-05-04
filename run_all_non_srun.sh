#!/bin/bash

SCENES=(
  09c1414f1b
  0d2ee665be
  38d58a7a31
  3db0a1c8f3
  5ee7c22ba0
  5f99900f09
  a8bf42d646
  a980334473
  c5439f4607
  cc5237fd77
)
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export TCNN_CUDA_ARCHITECTURES=90
export  LD_LIBRARY_PATH=/scratch/runyi_yang/miniconda3/envs/fmgs/lib:$LD_LIBRARY_PATH
conda init --all
conda activate fmgs

for SCENE in "${SCENES[@]}"; do
  echo "Processing scene: $SCENE"
  python train.py -s /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/$SCENE --dataformat scannetpp --model_path /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/$SCENE/gsplat/ --checkpoint_iterations 7000 30000 --port 6019

#   python train.py -s /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/$SCENE --dataformat scannetpp --model_path /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/$SCENE/gsplat/ --opt_vlrenderfeat_from 30000 --test_iterations 32000 32500 --save_iterations 32000 32500 --iterations 32500 --checkpoint_iterations 32000 32500 --start_checkpoint /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/$SCENE/gsplat/chkpnt30000.pth --fmap_resolution 2 --lambda_clip 0.2 --fmap_lr 0.005 --fmap_render_radiithre 2 --port 6019

done
# python train.py -s /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/09c1414f1b --dataformat scannetpp --model_path /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/09c1414f1b/gsplat/ 

# python train.py -s /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/09c1414f1b --dataformat scannetpp --model_path /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/09c1414f1b/gsplat/ --opt_vlrenderfeat_from 30000 --test_iterations 32000 32500 --save_iterations 32000 32500 --iterations 32500 --checkpoint_iterations 32000 32500 --start_checkpoint /home/runyi_yang/benchmark2025/dataset/ScanNetpp/data/09c1414f1b/gsplat/chkpnt30000.pth --fmap_resolution 2 --lambda_clip 0.2 --fmap_lr 0.005 --fmap_render_radiithre 2 --port 6009 