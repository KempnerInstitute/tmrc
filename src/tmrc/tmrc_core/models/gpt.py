import torch

import torch.nn as nn
from torch.nn import functional as F
from tmrc.tmrc_core.models.components import decoder

from tmrc.tmrc_core.utils import platform

from tmrc.tmrc_core.utils.registry import register_model

@register_model("gpt")
class GPT(nn.Module):
    """
    Basic implementation of GPT-ish variant architectures from OpenAI.
    This class follows Karpathy's implementation almost exactly. Minor
    changes were made to:
        - validate config
        - simplify optimization, fuse optimizer step into backward pass [TODO]
        - simplify overall class, e.g., no longer use GPT weight init
        - using the `Platform` class to manage device training
        - move FlashAttention -> FlexAttention [TODO]
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
            ln_f = nn.LayerNorm(config.model.d_model, bias=config.model.ln_bias),
        ))
        self.lm_head = nn.Linear(config.model.d_model, config.tokenizer.vocab_size, bias=config.model.cls_head_bias)
        self.loss_criterion = nn.CrossEntropyLoss()

        self.arange_T = torch.arange(0, config.model.context_length, dtype=torch.long)
        if self.platform.is_gpu:
            self.arange_T = self.platform.move_to_device(self.arange_T, device_index=0)

        if self.config.flex:
            flex_attention = torch.compile(flex_attention, dynamic=False)

    @staticmethod
    def validate_config(config):
        """Some basic sanity checks for the model config."""

        assert config.tokenizer.vocab_size is not None, "valid vocabulary size not defined"
        assert config.model.context_length is not None, "context length must be valid int > 0"
        assert config.model.d_model % config.model.n_head == 0, "d_model must be divisible by n_head"

        assert (config.model.flash and config.model.flex) is not True, "flash and flex attention cannot be used simultaneously"

    def get_num_params(self, non_embedding=False):
        """Get total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None, block_mask = None):
        
        B, T = idx.shape
        assert T <= self.config.model.context_length, f"Sequence length {T} > context length {self.config.model.context_length}"

        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(self.arange_T) # (B, T, C)
        x = tok_emb + pos_emb
        if self.config.flex:
            assert block_mask is not None, "Flex attention requires block mask when calling forward"
            x = block(x, block_mask)
        else:
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.loss_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss