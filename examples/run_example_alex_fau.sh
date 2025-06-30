#!/bin/bash

#SBATCH --job-name=dalia
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
# ##SBATCH --qos=a100multi
# ##SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=2

backend=cupy
export ARRAY_MODULE=${backend}

export MPI_CUDA_AWARE=0
export USE_NCCL=0

TIMESTAMP=$(date +"%H-%M-%S")

# --- How to Run ---
# This run script is designed to run on Alex at NHR@FAU
# It uses SLURM for job scheduling and assumes that the user has a working 
# installation of pyINLA and its dependencies. By default, PyINLA will exploit  
# job parallelism at the parallel function evaluation level.

# --- Parameters ---
# `--solver_min_p` : The minimum number of Processes(/GPUs) to use for the structured 
#                    solver. The default is 1. The maximum number of processes is
# `--max_iter` : The maximum number of iterations of the minimization.

base_dir=~

# --- Run Regression Example ---
srun python ${base_dir}/pyINLA/examples/gr/run.py --max_iter 100

# --- Run Spatial Examples ---
srun python ${base_dir}/pyINLA/examples/gs_small/run.py --max_iter 100

# --- Run Spatio-temporal Examples ---
srun python ${base_dir}/pyINLA/examples/gst_small/run.py --solver_min_p 1 --max_iter 100
srun python ${base_dir}/pyINLA/examples/gst_medium/run.py --solver_min_p 1 --max_iter 100
srun python ${base_dir}/pyINLA/examples/gst_large/run.py --solver_min_p 1 --max_iter 100

# --- Run Coregional (Spatial) Examples ---
srun python ${base_dir}/pyINLA/examples/gs_coreg2_small/run.py --max_iter 100
srun python ${base_dir}/pyINLA/examples/gs_coreg3_small/run.py --max_iter 100

# --- Run Coregional (Spatio-temporal) Examples ---
srun python ${base_dir}/pyINLA/examples/gst_coreg2_small/run.py --solver_min_p 1 --max_iter 100
srun python ${base_dir}/pyINLA/examples/gst_coreg3_small/run.py --solver_min_p 1 --max_iter 100

# --- Run Poisson Examples ---
srun python ${base_dir}/pyINLA/examples/pr/run.py --max_iter 100
srun python ${base_dir}/pyINLA/examples/pst_small/run.py --max_iter 100
