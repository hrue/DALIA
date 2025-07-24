#!/bin/bash
# DALIA ALPS Environment Configuration
# Autonomous module loading for CSCS ALPS supercomputer
# 
# This file can be sourced by any DALIA script running on ALPS
# to ensure consistent environment setup without external dependencies.

# Function to setup uenv session
install_alps_uenv() {
    # Documentation: https://docs.cscs.ch/software/uenv/
    echo "🔧 Downloading ALPS prgenv..."
    
    # Check if the prgenv-gnu/24.11:v1 image is already pulled
    if uenv image list | grep -q "prgenv-gnu/24.11:v1"; then
        echo "   prgenv-gnu/24.11:v1 image already exists, skipping pull"
    else
        echo "   Pulling prgenv-gnu/24.11:v1 image..."
        uenv image pull prgenv-gnu/24.11:v1 || {
            echo "❌ Error: Failed to pull prgenv-gnu/24.11:v1 image"
            return 1
        }
    fi
}

start_alps_uenv() {
    # Documentation: https://docs.cscs.ch/software/uenv/
    echo "🔧 Starting ALPS prgenv..."
    
    # Stop any existing uenv session
    uenv stop 2>/dev/null || echo "   (No existing uenv to stop)"
    
    # Start new uenv session
    echo "   Starting uenv with prgenv-gnu/24.11:v1..."
    echo "   WARNING: This is gonna start a new shell session, if you want to"
    echo "   use other functions from this script, you need to source it again."
    uenv start --view=modules prgenv-gnu/24.11:v1
}

# Function to load ALPS modules with error handling
load_alps_modules() {
    echo "🔧 Loading DALIA environment on ALPS..."
    
    # Check if we're already in a uenv session
    if ! uenv status &>/dev/null; then
        echo "❌ Error: Not in a uenv session. Modules can only be loaded within a uenv session."
        return 1
    fi
    
    # Purge any existing modules
    module purge 2>/dev/null
    
    # Load required modules (excluding python to avoid conflicts with conda)
    echo "   Loading system modules (excluding Python to preserve conda environment)..."
    module load cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack || {
        echo "❌ Error: Failed to load required modules"
        echo "   Required modules: cuda, gcc, meson, ninja, nccl, cray-mpich, cmake, openblas, aws-ofi-nccl, netlib-scalapack"
        echo "   Note: python/3.12.5 module excluded to preserve conda environment"
        echo "   Available modules:"
        module avail 2>&1 
        return 1
    }

    # NCCL environment setup
    export NCCL_ROOT=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd
    export NCCL_LIB_DIR=$NCCL_ROOT/lib
    export NCCL_INCLUDE_DIR=$NCCL_ROOT/include

    # CUDA environment setup
    if [[ -z "$CUDA_HOME" ]]; then
        # CUDA_HOME not set
        echo "❌ Error: CUDA_HOME not set"
        echo "   Please ensure the CUDA module properly sets CUDA_HOME"
        echo "   or manually set CUDA_HOME to your CUDA installation directory"
        return 1
    fi
    
    export CUDA_DIR=$CUDA_HOME
    export CUDA_PATH=$CUDA_HOME
    export CPATH=$CUDA_HOME/include:$CPATH
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CPATH=$NCCL_ROOT/include:$CPATH
    export CFLAGS="-I$NCCL_ROOT/include $CFLAGS"
    export LDFLAGS="-L$NCCL_ROOT/lib $LDFLAGS"
    export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
    
    # Verify critical modules are loaded
    echo "✅ System modules loaded successfully:"
    echo "   - GCC: $(gcc --version | head -n1)"
    echo "   - CUDA: $(nvcc --version 2>/dev/null | grep 'release' || echo 'CUDA not found')"
    echo "   - CUDA_HOME: ${CUDA_HOME:-'Not set'}"
    echo "   - SLURM srun: $(which srun)"
    echo "   Note: Python will be provided by conda environment and not by system modules"
    echo ""

    return 0
}

# Function to activate DALIA conda environment
activate_conda_env() {
    conda deactivate

    local env_name=${1:-allin}
    echo "   Activating conda environment '${env_name}'..."
    conda activate ${env_name} || {
        echo "❌ Error: Failed to activate conda environment '${env_name}'"
        echo "   Please ensure the '${env_name}' conda environment exists"
        echo "   Create with: conda create -n ${env_name} python=3.12"
        return 1
    }
    
    # Ensure conda Python takes precedence over system modules
    echo "   Ensuring conda Python takes precedence..."
    export PATH="$CONDA_PREFIX/bin:$PATH"
    
    # Verify the correct Python is being used
    local python_path=$(which python)
    if [[ "$python_path" == *"$CONDA_PREFIX"* ]]; then
        echo "✅ Conda environment '${env_name}' activated correctly"
        echo "   Python: $python_path"
    else
        echo "⚠️  Warning: System Python may still take precedence over conda Python"
        echo "   Current Python: $python_path"
        echo "   Expected: $CONDA_PREFIX/bin/python"
    fi
    echo ""
    
    return 0
}

# Main function to set up complete DALIA environment
setup_dalia_alps_environment() {
    echo "  Setting up DALIA environment on ALPS  "
    echo "========================================"
    
    # Load system modules
    load_alps_modules || return 1
    
    # Activate conda environment  
    activate_conda_env ${conda_env} || return 1
    
    echo ""
    echo "✅ DALIA ALPS environment ready."
    echo ""
    
    return 0
}



