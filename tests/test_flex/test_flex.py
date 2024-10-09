from olmo.data import  MemMapDataset, DataCollator, IterableDataset
import torch.distributed as dist
from olmo.torch_util import barrier, get_global_rank, get_world_size

from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from glob import glob
from pathlib import Path
import torch
import itertools

import numpy as np

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)

from tmrc.tmrc_core.data.utils import build_memmap_dataset, build_train_dataloader, move_to_device


data_paths = ["/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/0/part-0-00000.npy"]#glob("/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/**/*.npy")

def make_loader(paths):
    train_loader = build_train_dataloader(
        global_train_batch_size = 4,
        device_train_batch_size = 4 // get_world_size(),
        pad_direction = "right",
        pad_token_id = 1,
        max_seq_len = 2048, 
        memmap_dtype = getattr(np, "uint16"),  
        eos_token_id = 2, 
        paths = paths,
        save_folder = "./temp/",
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

def test_loader():
    train_loader = make_loader(data_paths)
    print(type(train_loader))
    
    sample = next(itertools.islice(train_loader, 1, None))
    print(sample)

    assert isinstance(train_loader, DataLoader)

    # for idx, batch in enumerate(train_loader):
    #     if idx == 3:
    #         batch = move_to_device(batch, "cuda")
    #         input_ids=batch["input_ids"],
    #         attention_mask=batch.get("attention_mask"),
    #         attention_bias=batch.get("attention_bias"),
    #         doc_lens=batch.get("doc_lens"),
    #         print("DOCUMENT LENGTHS: ")
    #         print(doc_lens)


# check block/causal mask somehow
