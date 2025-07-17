#!/usr/bin/env python3
# Copyright 2024-2025 DALIA authors. All rights reserved.

"""
Simple test runner for communicator tests that can be run without pytest.
This is useful for quick verification during development.
"""

import sys
import traceback
from dalia import backend_flags
from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import CommunicatorConfig

if backend_flags["mpi_avail"]:
    from mpi4py import MPI


class TestCommunicator(Communicator):
    """Test class that implements methods for demonstration purposes."""
    def allreduce(self): return "test_allreduce"
    def allgather(self): return "test_allgather"  
    def allgatherv(self): return "test_allgatherv"
    def alltoall(self): return "test_alltoall"
    def bcast(self): return "test_bcast"
    def barrier(self, sync_gpu: bool = False): return "test_barrier"
    def split(self): return "test_split"


def test_basic_functionality():
    """Test basic communicator functionality."""
    print("Testing basic functionality...")
    
    # Test that Communicator can be instantiated directly (no longer abstract)
    comm = Communicator()
    assert hasattr(comm, 'rank')
    assert hasattr(comm, 'size')
    assert hasattr(comm, 'effective_comm_lib')
    assert comm.effective_comm_lib in ["nccl", "device_mpi", "host_mpi", "none"]
    print(f"✓ Concrete Communicator works, using: {comm.effective_comm_lib}")
    
    # Test none configuration
    config = CommunicatorConfig(comm_lib="none")
    comm = Communicator(config)
    assert comm.effective_comm_lib == "none"
    assert comm.rank == 0
    assert comm.size == 1
    assert comm.mpi_comm is None
    print("✓ 'none' comm_lib works")
    
    # Test that methods raise NotImplementedError
    try:
        comm.allreduce()
        print("✗ allreduce should raise NotImplementedError")
        return False
    except NotImplementedError:
        print("✓ allreduce properly raises NotImplementedError")
    except Exception as e:
        print(f"✗ allreduce raised unexpected error: {e}")
        return False
    
    return True


def test_backend_availability():
    """Test backend availability validation."""
    print("\nTesting backend availability...")
    
    backends_to_test = [
        ("host_mpi", backend_flags["mpi_avail"]),
        ("device_mpi", backend_flags["mpi_cuda_aware"]),
        ("nccl", backend_flags["nccl_avail"])
    ]
    
    for backend, available in backends_to_test:
        config = CommunicatorConfig(comm_lib=backend)
        
        if available:
            try:
                comm = Communicator(config)
                print(f"✓ {backend} backend works (available)")
            except Exception as e:
                print(f"✗ {backend} backend failed despite being available: {e}")
                return False
        else:
            try:
                comm = Communicator(config)
                print(f"✗ {backend} backend should have failed (not available)")
                return False
            except RuntimeError:
                print(f"✓ {backend} backend correctly failed (not available)")
            except Exception as e:
                print(f"✗ {backend} backend failed with unexpected error: {e}")
                return False
    
    return True


def test_invalid_backend():
    """Test invalid backend handling at both Pydantic and custom validation levels."""
    print("\nTesting invalid backend handling...")
    
    # Test 1: Pydantic validation catches completely invalid values at config creation
    try:
        config = CommunicatorConfig(comm_lib="invalid_backend")
        print("✗ Invalid backend should have failed at config creation")
        return False
    except ValueError as e:
        # Pydantic validation error - this is expected and good
        if "literal_error" in str(e) or "Input should be" in str(e):
            print("✓ Invalid backend correctly rejected by Pydantic validation")
        else:
            print(f"✗ Invalid backend failed with unexpected Pydantic error: {e}")
            return False
    except Exception as e:
        print(f"✗ Invalid backend failed with unexpected error type: {e}")
        return False

    # Test 2: Custom validation catches valid config values that aren't available on system
    # We'll test this by trying a backend that's valid in config but not available
    # Since we know device_mpi and nccl are not available in this environment
    
    unavailable_backends = []
    if not backend_flags["mpi_cuda_aware"]:
        unavailable_backends.append("device_mpi")
    if not backend_flags["nccl_avail"]:
        unavailable_backends.append("nccl")
    
    if unavailable_backends:
        test_backend = unavailable_backends[0]
        config = CommunicatorConfig(comm_lib=test_backend)
        try:
            comm = Communicator(config)
            print(f"✗ {test_backend} backend should have failed (not available)")
            return False
        except RuntimeError as e:
            print(f"✓ {test_backend} backend correctly rejected by custom validation")
        except Exception as e:
            print(f"✗ {test_backend} backend failed with unexpected error: {e}")
            return False
    else:
        print("✓ No unavailable backends to test custom validation (all backends available)")
    
    return True


def test_mpi_functionality():
    """Test MPI-specific functionality."""
    if not backend_flags["mpi_avail"]:
        print("\nSkipping MPI tests (MPI not available)")
        return True
        
    print("\nTesting MPI functionality...")
    
    config = CommunicatorConfig(comm_lib="host_mpi")
    comm = Communicator(config)
    
    # Check MPI attributes
    assert comm.mpi_comm is not None
    assert isinstance(comm.rank, int)
    assert isinstance(comm.size, int)
    assert comm.rank >= 0
    assert comm.size >= 1
    assert comm.rank < comm.size
    
    # Check consistency with MPI.COMM_WORLD
    world_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    assert comm.rank == world_rank
    assert comm.size == world_size
    
    print(f"✓ MPI functionality works (rank {comm.rank}/{comm.size})")
    return True


def test_priority_resolution():
    """Test default priority resolution."""
    print("\nTesting priority resolution...")
    
    config = CommunicatorConfig(comm_lib="default")
    comm = Communicator(config)
    
    # Check expected priority order
    if backend_flags["nccl_avail"]:
        expected = "nccl"
    elif backend_flags["mpi_cuda_aware"]:
        expected = "device_mpi"
    elif backend_flags["mpi_avail"]:
        expected = "host_mpi"
    else:
        expected = "none"
    
    if comm.effective_comm_lib == expected:
        print(f"✓ Priority resolution works, selected: {expected}")
        return True
    else:
        print(f"✗ Priority resolution failed, expected {expected}, got {comm.effective_comm_lib}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DALIA Communicator Test Runner")
    print("=" * 60)
    
    print(f"Backend flags:")
    print(f"  MPI available: {backend_flags['mpi_avail']}")
    print(f"  MPI CUDA-aware: {backend_flags['mpi_cuda_aware']}")
    print(f"  NCCL available: {backend_flags['nccl_avail']}")
    print(f"  CuPy available: {backend_flags['cupy_avail']}")
    
    tests = [
        test_basic_functionality,
        test_backend_availability,
        test_invalid_backend,
        test_mpi_functionality,
        test_priority_resolution,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        test_name = test.__name__
        try:
            result = test()
            if result:
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! 🎉")


if __name__ == "__main__":
    main()
