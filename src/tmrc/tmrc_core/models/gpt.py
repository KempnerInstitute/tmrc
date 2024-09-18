import torch

import torch.nn as nn
from torch.nn import functional as F
from .components import decoder

from ..utils import platform, registry
#from utils.platform import Platform, auto_device


class GPT(nn.Module):
    """
    Basic implementation of GPT-ish variant architectures from OpenAI.
    This class follows Karpathy's implementation almost exactly. Minor
    changes were made to:
        - validate config
        - simplify optimization, fuse optimizer step into backward pass [TO DO]
        - simplify overall class, e.g., no longer use GPT weight init
        - using the `Platform` class to manage device training
        - move FlashAttention -> FlexAttention [TO DO]
    """

    def __init__(self, config, platform: platform.Platform):
        super().__init__()
        GPT.validate_config(config)

        self.config = config
        self.platform = platform

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.tokenizer.vocab_size, config.model.d_model),
            wpe = nn.Embedding(config.model.context_length, config.model.d_model),
            drop = nn.Dropout(config.model.dropout_p),
            h = nn.ModuleList([decoder.Block(config) for _ in range(config.model.n_layer)]),
            ln_f = nn.LayerNorm(config.model.d_model, bias=config.model.bias),
        ))
        self.lm_head = nn.Linear(config.model.d_model, config.tokenizer.vocab_size, bias=False)
        self.loss_criterion = nn.CrossEntropyLoss()

    @staticmethod
    def validate_config(config):
        """Some basic sanity checks for the model config."""

        assert config.tokenizer.vocab_size is not None
        assert config.model.context_length is not None
        assert config.model.d_model % config.model.n_head == 0, "d_model must be divisible by n_head"

    def get_num_params(self, non_embedding=False):
        """Get total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    @platform.auto_device
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        assert T <= self.config.model.context_length, f"Sequence length {T} > context length {self.config.model.context_length}"

        pos = torch.arange(0, T, dtype=torch.long) # (T)
        
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(pos) # (B, T, C)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.loss_criterion(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss