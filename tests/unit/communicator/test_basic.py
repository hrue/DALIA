# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest
from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import CommunicatorConfig


class TestCommunicatorBasic:
    """Basic unit tests for the Communicator class."""

    def test_communicator_is_concrete(self):
        """Test that Communicator can be instantiated directly (no longer abstract)."""
        config = CommunicatorConfig(comm_lib="none")
        # This should work now since Communicator is concrete
        comm = Communicator(config)
        assert comm.effective_comm_lib == "none"
        assert comm.rank == 0
        assert comm.size == 1

    def test_methods_raise_not_implemented_error(self):
        """Test that all communication methods raise NotImplementedError."""
        config = CommunicatorConfig(comm_lib="none")
        comm = Communicator(config)
        
        # Test that all methods raise NotImplementedError
        with pytest.raises(NotImplementedError, match="allreduce method not yet implemented"):
            comm.allreduce()
        
        with pytest.raises(NotImplementedError, match="allgather method not yet implemented"):
            comm.allgather()
            
        with pytest.raises(NotImplementedError, match="allgatherv method not yet implemented"):
            comm.allgatherv()
            
        with pytest.raises(NotImplementedError, match="alltoall method not yet implemented"):
            comm.alltoall()
            
        with pytest.raises(NotImplementedError, match="bcast method not yet implemented"):
            comm.bcast()
            
        with pytest.raises(NotImplementedError, match="barrier method not yet implemented"):
            comm.barrier()
            
        with pytest.raises(NotImplementedError, match="split method not yet implemented"):
            comm.split()

    def test_default_config_initialization(self):
        """Test initialization with default configuration."""
        comm = Communicator()
        
        # Check that config is properly set
        assert isinstance(comm.config, CommunicatorConfig)
        assert comm.config.comm_lib == "default"
        
        # Check that effective_comm_lib is resolved
        assert comm.effective_comm_lib in ["nccl", "device_mpi", "host_mpi", "none"]
        
        # Check basic attributes
        assert hasattr(comm, 'rank')
        assert hasattr(comm, 'size')
        assert hasattr(comm, 'tag')
        assert comm.tag is None

    def test_invalid_comm_lib_raises_error(self):
        """Test that invalid communication library raises ValueError."""
        # Test that Pydantic catches invalid values at config level
        with pytest.raises(ValueError, match="Input should be"):
            config = CommunicatorConfig(comm_lib="invalid_lib")

    def test_none_comm_lib_single_process_mode(self):
        """Test single process mode when comm_lib is 'none'."""
        config = CommunicatorConfig(comm_lib="none")
        comm = Communicator(config)
        
        assert comm.effective_comm_lib == "none"
        assert comm.mpi_comm is None
        assert comm.rank == 0
        assert comm.size == 1

    def test_config_preservation(self):
        """Test that the original config is preserved."""
        config = CommunicatorConfig(
            comm_lib="default",
            allreduce="nccl",
            allgather="host_mpi"
        )
        comm = Communicator(config)
        
        # Check that original config is preserved
        assert comm.config == config
        assert comm.config.allreduce == "nccl"
        assert comm.config.allgather == "host_mpi"
