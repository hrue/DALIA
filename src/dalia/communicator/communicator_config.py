# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict

class CommunicatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Communication library to use
    # - "default" will use the best available option (nccl > device_mpi > host_mpi > none)
    # - "none" disables communication (single process mode)
    comm_lib: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"

    # Specific Communication Module to use per Collective
    # - "default" will use the given comm_lib
    allreduce: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    allgather: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    allgatherv: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    alltoall: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    bcast: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"

class FCommunicatorConfig(CommunicatorConfig):
    """
    Configuration for the communicator using the default backend.
    This is a placeholder for future extensions or specific configurations.
    """
    pass

class QCommunicatorConfig(CommunicatorConfig):
    """
    Configuration for the communicator using the default backend.
    This is a placeholder for future extensions or specific configurations.
    """
    pass

class SCommunicatorConfig(CommunicatorConfig):
    """
    Configuration for the communicator using the default backend.
    This is a placeholder for future extensions or specific configurations.
    """
    pass



class DALIACommunicatorConfig(BaseModel):
    """Container for all communicator configurations."""
    model_config = ConfigDict(extra="forbid")
    
    base: CommunicatorConfig
    f: FCommunicatorConfig
    q: QCommunicatorConfig  
    s: SCommunicatorConfig

def parse_dalia_communicator_config(config: Union[dict, str, Path]) -> DALIACommunicatorConfig:
    """
    Parse communicator configuration from TOML file or dict.
    
    If only 'base' or 'communicator' is specified, its values will be used
    as defaults for f, q, and s communicators unless they are explicitly overridden.
    
    Example TOML:
    ```toml
    [communicator.base]
    comm_lib = "nccl"
    allgather = "device_mpi"
    
    [communicator.f]
    # Will inherit all from base
    
    [communicator.q]
    # Will inherit from base, but can override specific fields
    allgather = "host_mpi"
    allreduce = "host_mpi"
    
    [communicator.s]
    # Will inherit all from base
    ```
    """
    if isinstance(config, (str, Path)):
        with open(config, "rb") as f:
            config_dict = tomllib.load(f)
    else:
        config_dict = config.copy()
    
    comm_config = config_dict.get("communicator", {})
    
    # Check if this is a simple config (no nested sections)
    has_nested = any(key in comm_config for key in ["base", "f", "q", "s"])
    
    if not has_nested:
        # Simple config - use the same config for all
        base_config = CommunicatorConfig(**comm_config)
        base_dict = base_config.model_dump()
        
        return DALIACommunicatorConfig(
            base=base_config,
            f=FCommunicatorConfig(**base_dict),
            q=QCommunicatorConfig(**base_dict),
            s=SCommunicatorConfig(**base_dict)
        )
    else:
        # Advanced config with inheritance
        base_config_dict = comm_config.get("base", {})
        base_config = CommunicatorConfig(**base_config_dict)
        base_dict = base_config.model_dump()
        
        f_dict = {**base_dict, **comm_config.get("f", {})}
        q_dict = {**base_dict, **comm_config.get("q", {})}
        s_dict = {**base_dict, **comm_config.get("s", {})}
        
        return DALIACommunicatorConfig(
            base=base_config,
            f=FCommunicatorConfig(**f_dict),
            q=QCommunicatorConfig(**q_dict),
            s=SCommunicatorConfig(**s_dict)
        )