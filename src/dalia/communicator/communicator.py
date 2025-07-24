from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass

import time

import numpy as np

from dalia import xp, backend_flags
from dalia.communicator.communicator_config import CommunicatorConfig

if backend_flags["cupy_avail"]:
    import cupy as cp

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

    # Set the MPI operation mapping
    mpi_op = {
        "sum": MPI.SUM,
        "prod": MPI.PROD,
        "min": MPI.MIN,
        "max": MPI.MAX,
    }

    if backend_flags["nccl_avail"]:
        from cupy.cuda import nccl

        # Set the NCCL Datatype Mapping
        nccl_datatype = {
            np.float32: nccl.NCCL_FLOAT,
            cp.float32: nccl.NCCL_FLOAT,
            np.complex64: nccl.NCCL_FLOAT,
            cp.complex64: nccl.NCCL_FLOAT,
            np.float64: nccl.NCCL_DOUBLE,
            cp.float64: nccl.NCCL_DOUBLE,
            np.complex128: nccl.NCCL_DOUBLE,
            cp.complex128: nccl.NCCL_DOUBLE,
        }

        # Set the NCCL Operation Mapping
        nccl_op = {
            "sum": cp.cuda.nccl.NCCL_SUM,
            "prod": cp.cuda.nccl.NCCL_PROD,
            "min": cp.cuda.nccl.NCCL_MIN,
            "max": cp.cuda.nccl.NCCL_MAX,
        }


@dataclass
class CollectiveConfig:
    allreduce: str = "default"
    allgather: str = "default"
    allgatherv: str = "default"
    alltoall: str = "default"
    bcast: str = "default"
    reduce: str = "default"
    reduce_scatter: str = "default"
    send_recv: str = "default"


class Communicator:
    def __init__(
        self,
        config: CommunicatorConfig = CommunicatorConfig(),
        communicator=None,
    ):
        self.tag = config.tag

        # Validate the availability of the communication library and set most performant default
        self.general_comm_lib = self._resolve_and_validate_comm_lib(
            comm_lib=config.comm_lib
        )
        self.collective_config: CollectiveConfig = (
            self._resolve_and_validate_collective_config(config)
        )

        # Initialize the communicators
        self.base_comm, self.rank, self.size = self._initialize_base_communicator(
            communicator
        )
        self._xccl_comm = self._initialize_xccl_communicator()

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

    def _initialize_base_communicator(self, base_comm=None):
        # Initialize MPI communicator and attributes based on general comm lib
        if self.general_comm_lib in ["host_mpi", "device_mpi", "nccl"]:
            # All these require MPI as a base layer
            base_comm = base_comm if base_comm is not None else MPI.COMM_WORLD
            rank = base_comm.rank
            size = base_comm.size
        else:
            # No communication (single process mode)
            base_comm = None
            rank = 0
            size = 1

        return base_comm, rank, size

    def _initialize_xccl_communicator(self):
        if (
            backend_flags["nccl_avail"]
            and self.general_comm_lib == "nccl"
            or any(lib in self.collective_config.__dict__.values() for lib in ["nccl"])
        ):
            if self.rank == 0:
                nccl_id = cp.cuda.nccl.get_unique_id()
                self.base_comm.bcast(nccl_id, root=0)
            else:
                nccl_id = self.base_comm.bcast(None, root=0)

            # Initialize NCCL communicator
            nccl_comm = cp.cuda.nccl.NcclCommunicator(
                ndev=self.size,
                commId=nccl_id,
                rank=self.rank,
            )

            self.barrier()

            return nccl_comm
        else:
            # No NCCL communicator needed
            return None

    def _resolve_and_validate_collective_config(self, config: CommunicatorConfig):
        """Resolve 'default' collective comm libs to the best available option and validate availability."""
        collective_config: CollectiveConfig = CollectiveConfig()
        for key in [
            "allreduce",
            "allgather",
            "allgatherv",
            "alltoall",
            "bcast",
            "reduce",
            "reduce_scatter",
            "send_recv",
        ]:
            lib = config.__dict__.get(key, "default")
            if lib == "default":
                # Use the general comm_lib for default collectives
                resolved_lib = self.general_comm_lib
            else:
                # Validate and resolve the specific collective comm lib
                resolved_lib = self._resolve_and_validate_comm_lib(lib)
            collective_config.__dict__[key] = resolved_lib
        return collective_config

    def _get_dtype_factor(self, arr):
        """Get the factor for complex vs real arrays (complex arrays need factor of 2)."""
        return 2 if np.iscomplexobj(arr) else 1

    # Collectives
    def allreduce(self, arr, op):
        def _get_allreduce_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_allreduce_parameters(arr)

        self.barrier()

        if self.collective_config.allreduce == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self.base_comm.Allreduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=comm_arr,
                op=mpi_op[op],
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self.collective_config.allreduce == "device_mpi":
            self.base_comm.Allreduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=arr,
                op=mpi_op[op],
            )
        elif self.collective_config.allreduce == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.allReduce(
                sendbuf=arr.data.ptr,
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                op=nccl_op[op],
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self.collective_config.allreduce == "none":
            pass

        self.barrier()

    def allgather(self, arr):
        def _get_allgather_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = (arr.size // self.size) * factor
            displacement = count * self.rank * (arr.dtype.itemsize // factor)
            return count, displacement

        count, displacement = _get_allgather_parameters(arr)

        self.barrier()

        if self.collective_config.allgather == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self.base_comm.Allgather(
                sendbuf=comm_arr,
                recvbuf=comm_arr,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self.collective_config.allgather == "device_mpi":
            self.base_comm.Allgather(
                sendbuf=arr,
                recvbuf=arr,
            )
        elif self.collective_config.allgather == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.allGather(
                sendbuf=arr.data.ptr + displacement,
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self.collective_config.allgather == "none":
            pass

        self.barrier()

    def allgatherv(self):
        raise NotImplementedError("allgatherv method not yet implemented")

    def alltoall(self):
        raise NotImplementedError("alltoall method not yet implemented")

    def reduce(self, arr, op, root: int = 0):
        def _get_reduce_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_reduce_parameters(arr)

        self.barrier()

        if self.collective_config.reduce == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self.base_comm.Reduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=comm_arr,
                op=mpi_op[op],
                root=root,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self.collective_config.reduce == "device_mpi":
            self.base_comm.Reduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=arr,
                op=mpi_op[op],
                root=root,
            )
        elif self.collective_config.reduce == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.reduce(
                sendbuf=arr.data.ptr,
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                op=nccl_op[op],
                root=root,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self.collective_config.reduce == "none":
            pass

        self.barrier()

    def reduce_scatter(self):
        raise NotImplementedError("reduce_scatter method not yet implemented")

    def bcast(self, arr, root: int = 0):
        def _get_bcast_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_bcast_parameters(arr)

        self.barrier()

        if self.collective_config.allgather == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self.base_comm.Bcast(
                buf=comm_arr,
                root=root,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self.collective_config.allgather == "device_mpi":
            self.base_comm.Bcast(
                buf=arr,
                root=root,
            )
        elif self.collective_config.allgather == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.bcast(
                buff=arr.data.ptr,
                count=count,
                datatype=datatype,
                root=root,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self.collective_config.allgather == "none":
            pass

        self.barrier()

    def send(self, arr, dest, tag):
        def _get_send_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_send_parameters(arr)

        self.barrier()

        if self.collective_config.send_recv == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self.base_comm.Send(
                buf=comm_arr,
                dest=dest,
                tag=tag,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self.collective_config.send_recv == "device_mpi":
            self.base_comm.Send(
                buf=arr,
                dest=dest,
                tag=tag,
            )
        elif self.collective_config.send_recv == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.send(
                sendbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                peer=dest,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self.collective_config.send_recv == "none":
            pass

        self.barrier()

    def recv(self, arr, source, tag):
        def _get_recv_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_recv_parameters(arr)

        self.barrier()

        if self.collective_config.send_recv == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self.base_comm.Recv(
                buf=comm_arr,
                source=source,
                tag=tag,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self.collective_config.send_recv == "device_mpi":
            self.base_comm.Recv(
                buf=arr,
                source=source,
                tag=tag,
            )
        elif self.collective_config.send_recv == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.recv(
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                peer=source,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self.collective_config.send_recv == "none":
            pass

        self.barrier()

    # Utilities
    def barrier(self, sync_gpu: bool = True):
        if self.base_comm is not None:
            self.base_comm.Barrier()
        if sync_gpu and backend_flags["cupy_avail"]:
            cp.cuda.Stream.null.synchronize()

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
