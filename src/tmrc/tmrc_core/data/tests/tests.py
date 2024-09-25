from olmo.data import  MemMapDataset, DataCollator, IterableDataset
import torch.distributed as dist
from olmo.torch_util import barrier, get_global_rank, get_world_size

from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from glob import glob
from pathlib import Path
import torch

import numpy as np

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)

from .utils import build_memmap_dataset, build_train_dataloader, move_to_device

import hydra

# @#hydra.main(config_path="./test_config.yaml", config_name="test_gpt_config")
# def get_model_test(config: DictConfig):
#     model = gpt.GPT(config)
#     print(f"{model.__name__}")
#     assert 1==1


paths = ["/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/0/part-0-00000.npy"]#glob("/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/**/*.npy")

def make_loader(paths):
    train_loader = build_train_dataloader(
        global_train_batch_size = 8,
        device_train_batch_size = 8 // get_world_size(),
        pad_direction = "right",
        pad_token_id = 1,
        max_seq_len = 2048, 
        memmap_dtype = getattr(np, "uint16"),  
        eos_token_id = 2, 
        paths,

        save_folder "./temp/",
        num_workers = 16,
        pin_memory = True,
        prefetch_factor = 16,
        persistent_workers = True,
        timeout = 0,
        drop_last = True,
        save_overwrite = True,
    )
    return train_loader

# check for doc mask length match

# check block/causal mask somehow
