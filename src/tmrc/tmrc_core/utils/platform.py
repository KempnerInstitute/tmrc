from dataclasses import dataclass, field
import torch
import torch.distributed as dist

from functools import wraps
from typing import List, Optional, Callable, Union


@dataclass
class Platform:
    """A basic platform abstraction to manage distributed training.
    Or, if we are not running on multiple nodes, identifies the single
    GPU device. """
    
    devices: List[torch.device] = field(init=False)
    world_size: int = field(init=False)
    rank: int = field(init=False)
    distributed: bool = field(init=False)

    def __post_init__(self):
        self.devices = self._get_devices()
        self.world_size = len(self.devices)
        self.rank = 0
        self.distributed = False
        self._init_distributed()
    
    def _get_devices(self) -> List[torch.device]:
        if torch.cuda.is_available():
            return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            return [torch.device('cpu')]

    def _init_distributed(self):
        if self.world_size > 1:
            try:
                dist.init_process_group(backend='nccl')
                self.distributed = True
                self.rank = dist.get_rank()
            except Exception as e:
                print(f"Failed to initialize distributed training: {e}")

    @property
    def is_gpu(self) -> bool:
        return self.devices[0].type == 'cuda'

    @property
    def num_gpus(self) -> int:
        return len(self.devices) if self.is_gpu else 0

    def get_device(self, index: Optional[int] = None) -> torch.device:
        if index is None:
            index = self.rank
        return self.devices[index % len(self.devices)]

    def move_to_device(self, obj: torch.Tensor, device_index: Optional[int] = None) -> torch.Tensor:
        return obj.to(self.get_device(device_index))

    def get_memory_info(self, device_index: Optional[int] = None) -> dict:
        if self.is_gpu:
            device = self.get_device(device_index)
            return {
                'total': torch.cuda.get_device_properties(device).total_memory,
                'allocated': torch.cuda.memory_allocated(device),
                'cached': torch.cuda.memory_reserved(device)
            }
        else:
            return {}

    def synchronize(self):
        if self.distributed:
            dist.barrier()
        if self.is_gpu:
            torch.cuda.synchronize()
    
    def get_device_str(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_ids = ','.join([str(x) for x in range(torch.cuda.device_count())])
            return f'cuda:{device_ids}'
        else:
            return 'cpu'