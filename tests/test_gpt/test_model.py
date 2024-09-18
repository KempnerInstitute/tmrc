from tmrc.tmrc_core.models import gpt
from tmrc.tmrc_core.utils.platform import Platform
from hydra import compose, initialize
from omegaconf import DictConfig
import pytest
import torch
import torch.nn as nn

initialize(config_path=".", version_base=None)
config: DictConfig = compose(config_name="test_config")
platform = Platform()

@pytest.fixture(scope="module")
def model():
    return gpt.GPT(config, platform)

def test_model_creation():
    model = gpt.GPT(config, platform)
    assert isinstance(model, nn.Module)
    assert isinstance(model.transformer['wte'], nn.Embedding)
    assert isinstance(model.transformer['wpe'], nn.Embedding)
    assert isinstance(model.transformer['h'], nn.ModuleList)
    assert isinstance(model.transformer['ln_f'], nn.LayerNorm)

def test_num_parameters():
    pass

def test_forward_pass():
    pass

def test_backward_pass():
    pass