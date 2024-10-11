from olmo.data import  MemMapDataset, DataCollator, IterableDataset
import torch.distributed as dist
from olmo.torch_util import barrier, get_global_rank, get_world_size

from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from glob import glob
from pathlib import Path
import torch
import torch.nn as nn
import itertools

import numpy as np

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)

from tmrc.tmrc_core.data.utils import build_memmap_dataset, build_train_dataloader, move_to_device


from hydra import compose, initialize
from omegaconf import DictConfig
import pytest

from tmrc.tmrc_core.models import gpt
from tmrc.tmrc_core.utils.platform import Platform
from tmrc.tmrc_core.models.components import OPTIMIZER_REGISTRY
from tmrc.tmrc_core.models import MODEL_REGISTRY


initialize(config_path=".", version_base=None)
config: DictConfig = compose(config_name="test_flex")
platform = Platform()

@pytest.fixture(scope="module")
def model():
    return MODEL_REGISTRY.get(config.model.name)(config, platform)

def test_model_creation():
    model = MODEL_REGISTRY.get(config.model.name)(config, platform)
    assert isinstance(model, nn.Module)
    assert isinstance(model.transformer['wte'], nn.Embedding)
    assert isinstance(model.transformer['wpe'], nn.Embedding)
    assert isinstance(model.transformer['h'], nn.ModuleList)
    assert isinstance(model.transformer['ln_f'], nn.LayerNorm)

@pytest.fixture(scope="module")
def train_loader():
    data_paths = ["/n/holyscratch01/barak_lab/Lab/data/dolma-algebraic-stack-tokenized-llama/0/part-0-00000.npy"]
    train_loader = build_train_dataloader(
        global_train_batch_size = 4,
        device_train_batch_size = 4 // get_world_size(),
        pad_direction = "right",
        pad_token_id = 1,
        max_seq_len = 2048, 
        memmap_dtype = getattr(np, "uint16"),  
        eos_token_id = 2, 
        paths = data_paths,
        save_folder = "./temp/",
        num_workers = 8,
        pin_memory = True,
        prefetch_factor = 16,
        persistent_workers = True,
        timeout = 0,
        drop_last = True,
        save_overwrite = True,
    )
    return train_loader


def test_loader(train_loader):
    train_loader = train_loader
    assert isinstance(train_loader, DataLoader)

def test_mask(train_loader):
    sample = next(itertools.islice(train_loader, 3, None))
    print(sample)
    doc_lens = sample.get("doc_lens")

    doc_lens = doc_lens.masked_select(doc_lens != 0)
    doc_mask = torch.cat([torch.full([e.tolist()], i) for i, e in enumerate(doc_lens)]).reshape(sample["input_ids"].shape)
    print(doc_mask)
    mask_counts = doc_mask.unique(return_counts=True)[1]#.reshape(sample["doc_lens"].shape)
    # sample_counts = sample["doc_lens"]

    assert torch.equal(mask_counts, doc_lens)

    doc_mask = move_to_device(doc_mask, "cuda")
    print(doc_mask.shape)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask[b, q_idx] == doc_mask[b,kv_idx]
        return causal_mask & document_mask
            
            
    block_mask = create_block_mask(document_causal_mask, sample["input_ids"].shape[0], 1, sample["input_ids"].shape[-1], sample["input_ids"].shape[-1], device="cuda")
    print(f"\nBlock Mask:\n{block_mask}")

    print(block_mask.shape[:-1])
    print(sample["input_ids"].shape)

    # batch size
    assert block_mask.shape[0] == sample["input_ids"].shape[0]
    # seqlen
    assert block_mask.shape[-1] == sample["input_ids"].shape[-1]   


# test_loader()
# test_mask()
def test_forward_pass(model, train_loader):
    sample = next(itertools.islice(train_loader, 3, None))
    doc_lens = sample.get("doc_lens")
    doc_lens = doc_lens.masked_select(doc_lens != 0)
    doc_mask = torch.cat([torch.full([e.tolist()], i) for i, e in enumerate(doc_lens)]).reshape(sample["input_ids"].shape)
    doc_mask = move_to_device(doc_mask, "cuda")

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask[b, q_idx] == doc_mask[b,kv_idx]
        return causal_mask & document_mask
            
            
    block_mask = create_block_mask(document_causal_mask, sample["input_ids"].shape[0], 1, sample["input_ids"].shape[-1], sample["input_ids"].shape[-1], device="cuda")
    x = sample["input_ids"]
    if platform.is_gpu:
        x = platform.move_to_device(x, device_index=0)
        platform.move_to_device(model, device_index=0)

    output, _ = model(x, block_mask=block_mask)
    print(output.shape)
    print(output)
    assert output.shape == torch.Size([sample["input_ids"].shape[0], 1, config.tokenizer.vocab_size])

