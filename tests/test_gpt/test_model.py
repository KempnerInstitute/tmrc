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

def test_num_parameters(model):
    def expected_parameters_head(d_model):
        return 4*d_model**2+d_model

    def expected_parameters_block(d_model):
        return expected_parameters_head(d_model) + 8*d_model**2 + 5*d_model + 4*d_model # last is the 2 layer norms, each with 2 params

    def expected_parameters_total(vocab_size, d_model, n_layer, context_length):
        return n_layer*expected_parameters_block(d_model) + vocab_size*d_model + context_length*d_model + 2*d_model + d_model*vocab_size + vocab_size

    actual = sum(p.numel() for p in model.parameters())
    expected = expected_parameters_total(config.tokenizer.vocab_size, config.model.d_model, config.model.n_layer, config.model.context_length)
    assert actual==expected

def test_forward_pass():
    pass

def test_backward_pass():
    pass