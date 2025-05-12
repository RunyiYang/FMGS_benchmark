#!/bin/bash
#SBATCH --job-name=scannet6
#SBATCH --nodelist=dgx3
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:1         # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16      # Number of CPU cores per task
#SBATCH --time=1-0         # Job timeout
#SBATCH --output=output_logs/scannet6.log     # Redirect stdout to a log file
#SBATCH --error=output_logs/scannet6.err    # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email

conda init --all
source ~/.bashrc
source /scratch/runyi_yang/miniconda3/bin/activate
conda activate fmgs

# GPU/CUDA vars
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export TCNN_CUDA_ARCHITECTURES=90
export LD_LIBRARY_PATH=/scratch/runyi_yang/miniconda3/envs/fmgs/lib:$LD_LIBRARY_PATH

srun python train_scannet.py --start_idx 250 --end_idx 313