from tmrc.tmrc_core.models import gpt
from tmrc.tmrc_core.utils.platform import Platform
from tmrc.tmrc_core.models.components import OPTIMIZER_REGISTRY
from tmrc.tmrc_core.models import MODEL_REGISTRY
from hydra import compose, initialize
from omegaconf import DictConfig
import pytest
import torch
import torch.nn as nn

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

initialize(config_path=".", version_base=None)
config: DictConfig = compose(config_name="test_config")
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

def test_num_parameters(model):
    # note: this test is currently only configured for bias=True except attention bias
    def expected_parameters_head(d_model):
        return 4*d_model**2+d_model

    def expected_parameters_block(scale_factor, d_model):
        return expected_parameters_head(d_model) + 2*scale_factor*d_model**2 + \
              (scale_factor+1)*d_model + 4*d_model # last is the 2 layer norms, each with 2 params

    def expected_parameters_total(vocab_size, scale_factor, d_model, n_layer, context_length):
        return n_layer*expected_parameters_block(scale_factor, d_model) + \
              (vocab_size + context_length)*d_model + 2*d_model + d_model*vocab_size + vocab_size

    actual = sum(p.numel() for p in model.parameters())
    expected = expected_parameters_total(config.tokenizer.vocab_size,
                                         config.model.mlp_scale_factor,
                                         config.model.d_model,
                                         config.model.n_layer,
                                         config.model.context_length)
    assert actual==expected

def test_forward_pass(model):
    seed=42
    batch_size = 12
    x = torch.randint(low=0, high=config.tokenizer.vocab_size, size=(batch_size, config.model.context_length), dtype=torch.long)
    if platform.is_gpu:
        x = platform.move_to_device(x, device_index=0)
        platform.move_to_device(model, device_index=0)

    output, _ = model(x)
    assert output.shape == torch.Size([batch_size, 1, config.tokenizer.vocab_size])

def test_backward_pass(model):
    seed=42
    batch_size = 12
    x = torch.randint(low=0, high=config.tokenizer.vocab_size, size=(batch_size, config.model.context_length), dtype=torch.long)
    targets = torch.randint(low=0, high=config.tokenizer.vocab_size, size=(batch_size, config.model.context_length), dtype=torch.long)

    if platform.is_gpu:
        x = platform.move_to_device(x, device_index=0)
        targets = platform.move_to_device(targets, device_index=0)
        platform.move_to_device(model, device_index=0)
    
    print(x)
    print(targets)
    
    optimizer = OPTIMIZER_REGISTRY.get(config.optimizer.name)(params=model.parameters(),
                                                              lr=config.optimizer.lr,
                                                              betas=(config.optimizer.betas[0], config.optimizer.betas[1]),
                                                              eps=config.optimizer.eps,
                                                              weight_decay=config.optimizer.weight_decay)
    

    _, loss = model(x, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, new_loss = model(x, targets)
    assert new_loss.item() < loss