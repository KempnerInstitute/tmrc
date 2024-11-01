# Description: Test the flex attention mechanism in the TMRC model
import numpy as np
import pytest
import torch
import torch.nn as nn

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tatm.data import get_dataset, torch_collate_fn, TatmMemmapDataset
from tatm.tokenizer.metadata import write_metadata
from tmrc.tmrc_core.models import gpt, MODEL_REGISTRY
from tmrc.tmrc_core.models.components import OPTIMIZER_REGISTRY
from tmrc.tmrc_core.utils.platform import Platform
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    BlockMask
)
from torch.utils.data import DataLoader
from typing import Any, Dict, List

GlobalHydra.instance().clear()


initialize(config_path=".", version_base=None)
config: DictConfig = compose(config_name="test_flex")
platform = Platform()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


@pytest.fixture()
def sample_dataset(tmp_path):
    for i in range(10):
        data = np.memmap(
            tmp_path / f"test_{i}.bin", dtype="uint16", mode="w+", shape=(config.model.context_length,)
        )
        data[:] = i * config.model.context_length + np.arange(config.model.context_length)
        data.flush()
        del data
    write_metadata("t5-base", str(tmp_path), "test")
    yield (tmp_path, "test")

    for i in range(10):
        (tmp_path / f"test_{i}.bin").unlink()

def test_tatm_loader(sample_dataset):
    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), config.model.context_length, "uint16", create_doc_mask=True
    )
    tatm_dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=torch_collate_fn,
    )
    assert isinstance(tatm_dataloader, DataLoader)


def test_mask(sample_dataset):
    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), config.model.context_length, "uint16", create_doc_mask=True
    )
    tatm_dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=torch_collate_fn,
    )
    assert isinstance(tatm_dataloader, DataLoader)

    sample = next(iter(tatm_dataloader))
    doc_mask = sample.get("document_ids")
    doc_mask = doc_mask.to(torch.int32)
    doc_mask = doc_mask.to(device)

    x = sample["token_ids"].to(torch.int32)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask[b, q_idx] == doc_mask[b,kv_idx]
        return causal_mask & document_mask

    block_mask = create_block_mask(document_causal_mask, x.shape[0], None, x.shape[-1], x.shape[-1], device=device)
    assert isinstance(block_mask, BlockMask)

    # batch size
    assert block_mask.shape[0] == x.shape[0]
    # seqlen
    assert block_mask.shape[-1] == x.shape[-1]   


def test_forward(model, sample_dataset):
    dataset = TatmMemmapDataset(
        str(sample_dataset[0] / sample_dataset[1]), config.model.context_length, "uint16", create_doc_mask=True
    )
    tatm_dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=torch_collate_fn,
    )
    assert isinstance(tatm_dataloader, DataLoader)

    sample = next(iter(tatm_dataloader))
    doc_mask = sample.get("document_ids")
    doc_mask = doc_mask.to(torch.int32)
    doc_mask = doc_mask.to(device)

    x = sample["token_ids"].to(torch.int32)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask[b, q_idx] == doc_mask[b,kv_idx]
        return causal_mask & document_mask

    block_mask = create_block_mask(document_causal_mask, x.shape[0], 1, x.shape[-1], x.shape[-1], device=device)
    assert isinstance(block_mask, BlockMask)

    if platform.is_gpu:
        x = platform.move_to_device(x, device_index=0)
        platform.move_to_device(model, device_index=0)

    output, _ = model(x, doc_ids=doc_mask)
    assert output.shape == torch.Size([sample["token_ids"].shape[0], 1, config.tokenizer.vocab_size])



