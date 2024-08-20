import torch

import torch.nn as nn
from torch.nn import functional as F
from components import decoder


class GPT(nn.Module):
    """
    Basic implementation of GPT variant architectures from OpenAI.
    This class follows Karpathy's implementation almost exactly. Minor
    changes were made to:
        - validate config
        - simplify optimization, fuse optimizer step into backward pass
        - simplify overall class, e.g., no longer use GPT weight init
    """

    def __init__(self, config):
        super().__init__()
        GPT.validate_config(config)

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.ctx_len, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([decoder.Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @staticmethod
    def validate_config(config):
        """Some basic sanity checks for the model config."""

        assert config.vocab_size is not None
        assert config.context_length is not None
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"

    def get_num_params(self, non_embedding=False):
        """Get total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None):
        pass
