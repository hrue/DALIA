# Component Integration Tests

## Purpose

Component integration tests focus on testing **interactions within a single component** or **between closely related components**. These tests verify that different parts of a component work together correctly with real dependencies.

## What Goes Here

### ✅ **Good Examples:**
- Testing MPI communicator with actual MPI processes
- Testing GPU kernels with real CUDA operations  
- Testing file I/O with temporary files
- Testing backend selection with real hardware detection
- Testing component initialization with actual system resources
- Testing cross-backend compatibility matrices
- Testing environment consistency (e.g., MPI ranks, GPU devices)

### ❌ **What Doesn't Belong:**
- Full end-to-end framework workflows (→ `integration/`)
- Simple logic tests that don't need real dependencies (→ `unit/`)
- Tests involving multiple major components (→ `integration/`)

## Testing Pyramid Level

**Medium complexity, medium quantity:**
- More expensive than unit tests but cheaper than full integration
- Test actual system interactions within component boundaries
- May require specific hardware (MPI, GPU) but skip if unavailable
- Slower than unit tests (100ms - 5s per test typically)

## Structure

Each component has its own subdirectory with focused integration test files:

```
component_integration/
├── communicator/
│   ├── test_mpi_integration.py      # Real MPI functionality
│   ├── test_backend_matrix.py       # Cross-backend compatibility
│   └── test_gpu_communication.py    # CUDA-aware MPI testing
├── solvers/
│   ├── test_solver_convergence.py   # Numerical convergence
│   └── test_solver_scaling.py       # Performance characteristics  
└── kernels/
    ├── test_gpu_kernels.py          # Real GPU kernel execution
    └── test_kernel_accuracy.py      # Numerical accuracy testing
```

## Running Component Integration Tests

```bash
# Run all component integration tests
python -m pytest tests/component_integration/ -v

# Run for specific component
python -m pytest tests/component_integration/communicator/ -v

# Run with hardware requirements
python -m pytest tests/component_integration/ -v -m "not gpu" # Skip GPU tests
```

## Guidelines

1. **Use real dependencies** (MPI, GPU, files) when available
2. **Skip gracefully** when hardware/software is unavailable:
   ```python
   @pytest.mark.skipif(not backend_flags["mpi_avail"], reason="MPI not available")
   ```
3. **Test environment consistency** (ranks, devices, versions)
4. **Clean up resources** (files, communicators, GPU memory)
5. **Use fixtures** for common setup (MPI initialization, temporary directories)
6. **Test error conditions** with real system constraints

## Example Test Structure

```python
class TestCommunicatorMPIIntegration:
    """Integration tests for MPI functionality within communicator."""
    
    @pytest.mark.skipif(not backend_flags["mpi_avail"], reason="MPI not available")
    def test_mpi_allreduce_with_real_data(self):
        """Test MPI allreduce with actual MPI processes."""
        # Arrange
        config = CommunicatorConfig(comm_lib="host_mpi")
        comm = Communicator(config)
        data = np.array([comm.rank], dtype=np.float64)
        
        # Act
        result = comm.allreduce(data)
        
        # Assert
        expected_sum = sum(range(comm.size))
        assert result == expected_sum
        
    @pytest.fixture(autouse=True)
    def cleanup_mpi_resources(self):
        """Ensure MPI resources are properly cleaned up."""
        yield
        # Cleanup code here
```

## Hardware Requirements

Tests in this directory may require:
- **MPI**: Multiple processes, specific MPI implementations
- **GPU**: CUDA devices, specific GPU capabilities
- **Storage**: Temporary file access, specific file systems
- **Network**: Multi-node communication (for distributed tests)

Use `pytest.mark.skipif()` to handle missing requirements gracefully.
