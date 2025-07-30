#!/bin/bash
# DALIA ALPS Environment Configuration
# Autonomous module loading for CSCS ALPS supercomputer
# 
# This file can be sourced by any DALIA script running on ALPS
# to ensure consistent environment setup without external dependencies.

# Setup DALIA Performance Environment
set_alps_perfconfig() {

    set -e
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export MPICH_GPU_SUPPORT_ENABLED=1

    # NCCL Performance Configuration
    # More can be found: https://docs.cscs.ch/software/communication/nccl/#using-nccl
    export NCCL_NET='AWS Libfabric'
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1

    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_CXI_DEFAULT_TX_SIZE=32768
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_CXI_RX_MATCH_MODE=software
    export FI_MR_CACHE_MONITOR=userfaultfd
}

set_dalia_perfenv() {
    echo "🔧 Setting up DALIA performance environment..."
    
    # If we are in a SLURM job, call set_alps_perfconfig
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        set_alps_perfconfig
    else
        echo "   Not in a SLURM job, skipping ALPS specific performance environment setup"
    fi

    # Set DALIA environment variables
    export ARRAY_MODULE=cupy
    export MPI_CUDA_AWARE=1
    export USE_NCCL=1
    export MPICH_GPU_SUPPORT_ENABLED=0
}


# Print job configuration details and DALIA environment variables
echo_job_config() {
    echo "📋 SLURM Job Configuration:"
    echo "  - Job Name: ${SLURM_JOB_NAME}"
    echo "  - Job ID: ${SLURM_JOB_ID}"
    echo "  - Nodes: ${SLURM_NNODES}"
    echo "  - Tasks per node: ${SLURM_NTASKS_PER_NODE}"
    echo "  - Total tasks: ${SLURM_NTASKS}"
    echo "  - CPUs per task: ${SLURM_CPUS_PER_TASK}"
    echo "  - GPUs per task: 1"
    echo ""
    echo "🔧 DALIA Environment Configuration:"
    echo "  - ARRAY_MODULE: ${ARRAY_MODULE}"
    echo "  - MPI_CUDA_AWARE: ${MPI_CUDA_AWARE}"
    echo "  - USE_NCCL: ${USE_NCCL}"
    echo "  - MPICH_GPU_SUPPORT_ENABLED: ${MPICH_GPU_SUPPORT_ENABLED}"
    echo ""
}
