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

from utils import build_memmap_dataset, build_train_dataloader, move_to_device

flex_attention = torch.compile(flex_attention, dynamic = False)

global_train_batch_size = 1
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

    interest = 5
    sample = next(itertools.islice(train_loader, interest, None))
    print(sample)
    sample = move_to_device(sample, "cuda")
    input_ids = sample["input_ids"]

    for idx, batch in enumerate(train_loader):
        if idx == 5:
            batch = move_to_device(batch, "cuda")
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            print("DOCUMENT LENGTHS: ")
            print(doc_lens)
            max_doc_lens=batch.get("max_doc_lens"),

            doc_lens = doc_lens[0].masked_select(doc_lens[0] != 0)
            doc_mask = torch.cat([torch.full([e.tolist()], i) for i, e in enumerate(doc_lens)]).reshape(max_seq_len)

            doc_mask = move_to_device(doc_mask, "cuda")
            


            def document_causal_mask(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                document_mask = doc_mask[q_idx] == doc_mask[kv_idx]
                return causal_mask & document_mask
            
            block_mask = create_block_mask(document_causal_mask, 1, 1, max_seq_len, max_seq_len, device="cuda")
            print(f"\nBlock Mask:\n{block_mask}")
            print(block_mask.mask_mod)

            # flex_ms = flex_attention(
            #         query, key, value, score_mod=score_mod, block_mask=block_mask
            #     )

            # batch_doc_lens = doc_lens[0].masked_select(doc_lens[0] != 0)

            # print(batch_doc_lens)

            # batch_doc_mask = torch.cat([torch.full([e.tolist()], i) for i, e in enumerate(batch_doc_lens)]).reshape(device_train_batch_size, max_seq_len)
            # print(batch_doc_mask)
            # print(batch_doc_mask.unique(return_counts = True))
            # print(batch_doc_mask[0].unique(return_counts = True))




if __name__ == "__main__":
    main()