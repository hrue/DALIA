# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest
from unittest.mock import patch

from dalia import backend_flags
from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import CommunicatorConfig


class TestCommunicatorBackendSelection:
    """Tests for backend selection and validation logic."""

    @pytest.mark.parametrize("comm_lib", ["host_mpi", "device_mpi", "nccl", "none"])
    def test_specific_comm_lib_config(self, comm_lib):
        """Test initialization with specific communication library configurations."""
        config = CommunicatorConfig(comm_lib=comm_lib)
        
        # Test availability validation
        if comm_lib == "nccl" and not backend_flags["nccl_avail"]:
            with pytest.raises(RuntimeError, match="NCCL communication library was requested"):
                Communicator(config)
        elif comm_lib == "device_mpi" and not backend_flags["mpi_cuda_aware"]:
            with pytest.raises(RuntimeError, match="CUDA-aware MPI was requested"):
                Communicator(config)
        elif comm_lib == "host_mpi" and not backend_flags["mpi_avail"]:
            with pytest.raises(RuntimeError, match="MPI communication library was requested"):
                Communicator(config)
        else:
            # Should succeed if the library is available
            comm = Communicator(config)
            assert comm.effective_comm_lib == comm_lib

    def test_resolve_default_priority_order(self):
        """Test that default resolution follows the correct priority order."""
        # Test with default config
        config = CommunicatorConfig(comm_lib="default")
        comm = Communicator(config)
        
        # Verify priority order: nccl > device_mpi > host_mpi > none
        if backend_flags["nccl_avail"]:
            expected = "nccl"
        elif backend_flags["mpi_cuda_aware"]:
            expected = "device_mpi"
        elif backend_flags["mpi_avail"]:
            expected = "host_mpi"
        else:
            expected = "none"
            
        assert comm.effective_comm_lib == expected

    @patch('dalia.communicator.communicator.backend_flags')
    def test_validate_comm_lib_availability_mock(self, mock_backend_flags):
        """Test validation logic with mocked backend flags."""
        # Test NCCL validation
        mock_backend_flags.__getitem__.side_effect = lambda key: {
            "nccl_avail": False,
            "mpi_cuda_aware": False,
            "mpi_avail": False,
        }.get(key, False)
        
        config = CommunicatorConfig(comm_lib="nccl")
        with pytest.raises(RuntimeError, match="NCCL communication library was requested"):
            Communicator(config)

        # Test device MPI validation
        mock_backend_flags.__getitem__.side_effect = lambda key: {
            "nccl_avail": False,
            "mpi_cuda_aware": False,
            "mpi_avail": True,
        }.get(key, False)
        
        config = CommunicatorConfig(comm_lib="device_mpi")
        with pytest.raises(RuntimeError, match="CUDA-aware MPI was requested"):
            Communicator(config)

        # Test host MPI validation
        mock_backend_flags.__getitem__.side_effect = lambda key: {
            "nccl_avail": False,
            "mpi_cuda_aware": False,
            "mpi_avail": False,
        }.get(key, False)
        
        config = CommunicatorConfig(comm_lib="host_mpi")
        with pytest.raises(RuntimeError, match="MPI communication library was requested"):
            Communicator(config)

    def test_communicator_with_fixture_backends(self, COMM_LIB):
        """Test communicator initialization with different backends from conftest fixtures."""
        config = CommunicatorConfig(comm_lib=COMM_LIB)
        
        # This should not raise an error since conftest only includes available backends
        comm = Communicator(config)
        assert comm.effective_comm_lib == COMM_LIB
        
        # Verify attributes are correctly set
        if COMM_LIB == "none":
            assert comm.mpi_comm is None
            assert comm.rank == 0
            assert comm.size == 1
        else:
            if backend_flags["mpi_avail"]:
                assert comm.mpi_comm is not None
                assert isinstance(comm.rank, int)
                assert isinstance(comm.size, int)
