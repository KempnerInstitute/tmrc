from olmo.data import MemMapDataset, DataCollator, IterableDataset
import torch.distributed as dist
from olmo.torch_util import barrier, get_global_rank, get_world_size

from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Any, Dict, List, Optional, cast, TypeVar
from glob import glob
from pathlib import Path
import torch

import numpy as np

T = TypeVar("T")


# Arguments messy until we decide what to keep and discard from olmo config format


def build_memmap_dataset(
    max_seq_len, memmap_dtype, pad_token_id, eos_token_id, paths: List[str], datasets: Optional[Dict[str, List[str]]] = None, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if paths:
        if datasets:
            raise Exception("paths is mutually exclusive with datasets")
        paths = paths
        for path in paths:
            metadata.append({"path": str(path)})
        print(len(metadata))
    elif datasets:
        paths = []
        for label in sorted(datasets.keys()):
            label_paths = datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise Exception("One of paths or datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=max_seq_len,
        memmap_dtype=memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        generate_attention_mask=False,
        generate_doc_lengths=True,
        instance_filter_config=None,
    )


def build_train_dataloader(
    #train_config: TrainConfig,
    device_train_batch_size,
    global_train_batch_size,
    pad_direction,
    pad_token_id,
    max_seq_len, 
    memmap_dtype,  
    eos_token_id, 
    paths: List[str],

    save_folder,
    num_workers,
    pin_memory,
    prefetch_factor,
    persistent_workers,
    timeout,
    epoch = 0,
    drop_last = False,


    
    
    *,
    save_overwrite = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=pad_direction, pad_token_id=pad_token_id
    )
    dataset = build_memmap_dataset(
        max_seq_len, memmap_dtype, pad_token_id, eos_token_id, paths, include_instance_metadata=include_instance_metadata
    )
    work_dir = Path(save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not save_overwrite:
            raise Exception(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    seed = 1324 #train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            global_train_batch_size,
            seed=seed,
            epoch=epoch or 0,
            shuffle=True,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            fs_local_rank=fs_local_rank,
            work_dir=work_dir,
        ),
        batch_size=device_train_batch_size,
        drop_last=drop_last,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=False if num_workers == 0 else persistent_workers,
        timeout=timeout,
    )


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o