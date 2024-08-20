from olmo.data import  MemMapDataset, DataCollator, IterableDataset
import torch.distributed as dist
from olmo.torch_util import barrier, get_global_rank, get_world_size

from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
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

from utils import build_memmap_dataset, build_train_dataloader


global_train_batch_size = 2
device_train_microbatch_size = 8
device_train_batch_size = global_train_batch_size // get_world_size()
pad_direction = "right"
paths = ["/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/0/part-0-00000.npy"]#glob("/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/**/*.npy")
num_workers = 16
drop_last = True
pin_memory = True
prefetch_factor = 16
persistent_workers = True
timeout = 0
generate_doc_lengths = True
pad_token_id = 1
max_seq_len = 2048
memmap_dtype = getattr(np, "uint16")
eos_token_id = 2
save_folder = "./temp/"

def main():
    train_loader = build_train_dataloader(
        device_train_batch_size,
        global_train_batch_size,
        pad_direction,
        pad_token_id,
        max_seq_len, 
        memmap_dtype,  
        eos_token_id, 
        paths,

        save_folder,
        num_workers,
        pin_memory,
        prefetch_factor,
        persistent_workers,
        timeout,
        drop_last,
        save_overwrite = True,
    )

    for idx, batch in enumerate(train_loader):
        if idx == 2:
            print(batch["input_ids"].shape)
            input_ids=batch["input_ids"],
            print(input_ids)
            attention_mask=batch.get("attention_mask"),
            print(attention_mask)
            attention_bias=batch.get("attention_bias"),
            print(attention_bias)
            doc_lens=batch.get("doc_lens"),
            print(doc_lens)
            max_doc_lens=batch.get("max_doc_lens"),
            print(max_doc_lens)

    batch_doc_lens = doc_lens[0].masked_select(doc_lens[0] != 0)

    print(batch_doc_lens)

    batch_doc_mask = torch.cat([torch.full([e.tolist()], i) for i, e in enumerate(batch_doc_lens)]).reshape(device_train_batch_size, max_seq_len)
    print(batch_doc_mask)

if __name__ == "__main__":
    main()