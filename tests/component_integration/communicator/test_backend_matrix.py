# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest
from dalia import backend_flags
from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import CommunicatorConfig


class TestCommunicatorBackendMatrix:
    """Cross-backend compatibility and matrix testing for Communicator."""

    def test_communicator_backend_availability_matrix(self):
        """Test that communicator properly handles different backend availability scenarios."""
        # Test matrix of what should work based on current backend availability
        test_cases = [
            ("none", True),  # Should always work
            ("default", True),  # Should always work (falls back to available or none)
        ]
        
        if backend_flags["mpi_avail"]:
            test_cases.append(("host_mpi", True))
        else:
            test_cases.append(("host_mpi", False))
            
        if backend_flags["mpi_cuda_aware"]:
            test_cases.append(("device_mpi", True))
        else:
            test_cases.append(("device_mpi", False))
            
        if backend_flags["nccl_avail"]:
            test_cases.append(("nccl", True))
        else:
            test_cases.append(("nccl", False))

        for comm_lib, should_succeed in test_cases:
            config = CommunicatorConfig(comm_lib=comm_lib)
            
            if should_succeed:
                # Should not raise an exception
                comm = Communicator(config)
                assert comm.effective_comm_lib in ["none", "host_mpi", "device_mpi", "nccl"]
            else:
                # Should raise RuntimeError
                with pytest.raises(RuntimeError):
                    Communicator(config)

    def test_default_resolution_correctness(self):
        """Test that default resolution picks the expected backend."""
        config = CommunicatorConfig(comm_lib="default")
        comm = Communicator(config)
        
        # Verify the expected priority order is respected
        if backend_flags["nccl_avail"]:
            assert comm.effective_comm_lib == "nccl"
        elif backend_flags["mpi_cuda_aware"]:
            assert comm.effective_comm_lib == "device_mpi"
        elif backend_flags["mpi_avail"]:
            assert comm.effective_comm_lib == "host_mpi"
        else:
            assert comm.effective_comm_lib == "none"

    def test_communicator_with_conftest_backends(self, COMM_LIB):
        """Test communicator with backends from conftest fixtures."""
        config = CommunicatorConfig(comm_lib=COMM_LIB)
        
        # This should work since conftest only includes available backends
        comm = Communicator(config)
        
        # Verify basic functionality
        assert comm.effective_comm_lib == COMM_LIB
        assert hasattr(comm, 'rank')
        assert hasattr(comm, 'size') 
        assert hasattr(comm, 'tag')
        
        # Test that all methods exist but raise NotImplementedError
        with pytest.raises(NotImplementedError):
            comm.allreduce()
        with pytest.raises(NotImplementedError):
            comm.allgather()
        with pytest.raises(NotImplementedError):
            comm.allgatherv()
        with pytest.raises(NotImplementedError):
            comm.alltoall()
        with pytest.raises(NotImplementedError):
            comm.bcast()
        with pytest.raises(NotImplementedError):
            comm.barrier()
        with pytest.raises(NotImplementedError):
            comm.split()
