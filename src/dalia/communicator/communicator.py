from contextlib import contextmanager
from functools import wraps

import time

from dalia import xp, backend_flags
from dalia.communicator.communicator_config import CommunicatorConfig

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

class Communicator:
    def __init__(
            self, 
            config: CommunicatorConfig = CommunicatorConfig(),
            mpi_comm=None,
        ): 
        self.config = config
        
        # Determine and validate the effective communication library to use
        self.effective_comm_lib = self._resolve_and_validate_comm_lib(config.comm_lib)
        
        # Initialize MPI communicator and attributes based on effective comm lib
        if self.effective_comm_lib in ["host_mpi", "device_mpi", "nccl"]:
            # All these require MPI as a base layer
            self.mpi_comm = mpi_comm if mpi_comm is not None else MPI.COMM_WORLD
            self.rank = self.mpi_comm.Get_rank()
            self.size = self.mpi_comm.Get_size()
        else:
            # No communication (single process mode)
            self.mpi_comm = None
            self.rank = 0
            self.size = 1
        
        self.tag = None  # Will be set by subclasses if needed
        
    def _resolve_and_validate_comm_lib(self, comm_lib: str) -> str:
        """Resolve 'default' comm_lib to the best available option and validate availability."""
        if comm_lib == "default":
            # Priority order: nccl > device_mpi > host_mpi > none
            if backend_flags["nccl_avail"]:
                return "nccl"
            elif backend_flags["mpi_cuda_aware"]:
                return "device_mpi"
            elif backend_flags["mpi_avail"]:
                return "host_mpi"
            else:
                return "none"
        else:
            # Validate that the specified comm_lib is actually available
            self._validate_comm_lib_availability(comm_lib)
            return comm_lib
    
    def _validate_comm_lib_availability(self, comm_lib: str) -> None:
        """Validate that the specified communication library is available on the system."""
        if comm_lib == "nccl":
            if not backend_flags["nccl_avail"]:
                raise RuntimeError(
                    "NCCL communication library was requested but is not available. "
                    "Ensure NCCL is installed and USE_NCCL=1 environment variable is set."
                )
        elif comm_lib == "device_mpi":
            if not backend_flags["mpi_cuda_aware"]:
                raise RuntimeError(
                    "CUDA-aware MPI was requested but is not available. "
                    "Ensure MPI is compiled with CUDA support and MPI_CUDA_AWARE=1 environment variable is set."
                )
        elif comm_lib == "host_mpi":
            if not backend_flags["mpi_avail"]:
                raise RuntimeError(
                    "MPI communication library was requested but is not available. "
                    "Ensure mpi4py is installed and MPI is properly configured."
                )
        elif comm_lib not in ["none"]:
            raise ValueError(
                f"Unknown communication library '{comm_lib}'. "
                "Valid options are: 'default', 'host_mpi', 'device_mpi', 'nccl', 'none'"
            )

    # Collectives
    def allreduce(self):
        raise NotImplementedError("allreduce method not yet implemented")
    
    def allgather(self):
        raise NotImplementedError("allgather method not yet implemented")

    def allgatherv(self):
        raise NotImplementedError("allgatherv method not yet implemented")

    def alltoall(self):
        raise NotImplementedError("alltoall method not yet implemented")

    def bcast(self):
        raise NotImplementedError("bcast method not yet implemented")

    # Utilities
    def barrier(self, sync_gpu: bool = False):
        raise NotImplementedError("barrier method not yet implemented")

    def split(self):
        raise NotImplementedError("split method not yet implemented")

    @contextmanager
    def time(self):
        """Context manager for timing code blocks.
        
        Usage:
            with communicator.time() as elapsed_time:
                # Your code here
                L = cholesky(A)
            print(f"Elapsed time: {elapsed_time.value} seconds")
            
        Or simply:
            with communicator.time():
                # Your code here
                L = cholesky(A)
        """
        self.barrier(sync_gpu=True)
        
        class ElapsedTime:
            def __init__(self):
                self.value = None
        
        elapsed_time = ElapsedTime()
        tic = time.perf_counter()
        
        try:
            yield elapsed_time
        finally:
            self.barrier(sync_gpu=True)
            toc = time.perf_counter()
            elapsed_time.value = toc - tic

    def timed(self, print_result: bool = True, label: str = None):
        """Decorator for timing function execution.
        
        Args:
            print_result: Whether to print the elapsed time automatically
            label: Optional label to include in the printed output
        
        Usage:
            @communicator.timed()
            def compute_something():
                return expensive_operation()
                
            @communicator.timed(label="Cholesky decomposition")
            def cholesky_decomp(A):
                return cholesky(A)
                
            @communicator.timed(print_result=False)
            def silent_function():
                # This won't print timing, but you can access it via function.elapsed_time
                return result
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.barrier(sync_gpu=True)
                tic = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.barrier(sync_gpu=True)
                    toc = time.perf_counter()
                    elapsed = toc - tic
                    
                    # Store elapsed time as an attribute of the wrapper function
                    wrapper.elapsed_time = elapsed
                    
                    if print_result:
                        if label:
                            print(f"{label}: {elapsed:.6f} seconds")
                        else:
                            print(f"{func.__name__}: {elapsed:.6f} seconds")
            
            return wrapper
        return decorator

    