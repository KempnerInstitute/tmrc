# Description: Test the flex attention mechanism in the TMRC model
from torch.utils.data import DataLoader
from typing import Any, Dict, List
import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
)


from hydra import compose, initialize
from omegaconf import DictConfig
import pytest
from tmrc.tmrc_core.models import gpt
from tmrc.tmrc_core.utils.platform import Platform
from tmrc.tmrc_core.models.components import OPTIMIZER_REGISTRY
from tmrc.tmrc_core.models import MODEL_REGISTRY

from tatm.data import get_dataset, torch_collate_fn

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()


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


def test_loader():
    data_paths = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1/tokenized/t5-base/arxiv"
    tatm_dataset = get_dataset(data_paths, context_length=2048)
    tatm_dataloader = DataLoader(tatm_dataset, batch_size=4, collate_fn=torch_collate_fn)
    assert isinstance(tatm_dataloader, DataLoader)

def test_mask():
    data_paths = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1/tokenized/t5-base/arxiv"
    tatm_dataset = get_dataset(data_paths, context_length=2048)
    tatm_dataloader = DataLoader(tatm_dataset, batch_size=4, collate_fn=torch_collate_fn)
    assert isinstance(tatm_dataloader, DataLoader)

    sample = next(iter(tatm_dataloader))
    doc_mask = sample.get("document_ids")
    doc_mask = doc_mask.to(torch.int32)
    doc_mask = doc_mask.to("cuda")

    x = sample["token_ids"].to(torch.int32)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask[b, q_idx] == doc_mask[b,kv_idx]
        return causal_mask & document_mask

    block_mask = create_block_mask(document_causal_mask, x.shape[0], 1, x.shape[-1], x.shape[-1], device="cuda")

    # batch size
    assert block_mask.shape[0] == x.shape[0]
    # seqlen
    assert block_mask.shape[-1] == x.shape[-1]   


def test_forward_pass(model):
    data_paths = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1/tokenized/t5-base/arxiv"
    tatm_dataset = get_dataset(data_paths, context_length=2048)
    tatm_dataloader = DataLoader(tatm_dataset, batch_size=4, collate_fn=torch_collate_fn)
    assert isinstance(tatm_dataloader, DataLoader)

    sample = next(iter(tatm_dataloader))
    doc_mask = sample.get("document_ids")
    doc_mask = doc_mask.to(torch.int32)
    doc_mask = doc_mask.to("cuda")

    x = sample["token_ids"].to(torch.int32)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask[b, q_idx] == doc_mask[b,kv_idx]
        return causal_mask & document_mask

    block_mask = create_block_mask(document_causal_mask, x.shape[0], 1, x.shape[-1], x.shape[-1], device="cuda")
    if platform.is_gpu:
        x = platform.move_to_device(x, device_index=0)
        platform.move_to_device(model, device_index=0)

    output, _ = model(x, block_mask=block_mask)
    assert output.shape == torch.Size([sample["token_ids"].shape[0], 1, config.tokenizer.vocab_size])
