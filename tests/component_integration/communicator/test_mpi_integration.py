# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest
from dalia import backend_flags
from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import CommunicatorConfig

if backend_flags["mpi_avail"]:
    from mpi4py import MPI


class TestCommunicatorMPIIntegration:
    """MPI-specific integration tests for Communicator functionality."""

    @pytest.mark.skipif(not backend_flags["mpi_avail"], reason="MPI not available")
    def test_mpi_communicator_initialization(self):
        """Test MPI communicator initialization when MPI is available."""
        config = CommunicatorConfig(comm_lib="host_mpi")
        comm = Communicator(config)
        
        # Check MPI attributes
        assert comm.mpi_comm is not None
        assert isinstance(comm.rank, int)
        assert isinstance(comm.size, int)
        assert comm.rank >= 0
        assert comm.size >= 1
        assert comm.rank < comm.size

    @pytest.mark.skipif(not backend_flags["mpi_avail"], reason="MPI not available")
    def test_custom_mpi_communicator(self):
        """Test initialization with custom MPI communicator."""
        # Create a custom communicator (duplicate of COMM_WORLD for testing)
        custom_comm = MPI.COMM_WORLD.Dup()
        
        config = CommunicatorConfig(comm_lib="host_mpi")
        comm = Communicator(config, mpi_comm=custom_comm)
        
        assert comm.mpi_comm == custom_comm
        assert comm.rank == custom_comm.Get_rank()
        assert comm.size == custom_comm.Get_size()
        
        # Clean up
        custom_comm.Free()

    @pytest.mark.skipif(not backend_flags["mpi_avail"], reason="MPI not available")
    def test_mpi_environment_consistency(self):
        """Test that MPI environment is consistent across the test."""
        config = CommunicatorConfig(comm_lib="host_mpi")
        comm = Communicator(config)
        
        # Test that MPI world size and rank are consistent
        world_rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
        
        assert comm.rank == world_rank
        assert comm.size == world_size
