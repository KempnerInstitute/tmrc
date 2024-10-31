import torch

import torch.nn as nn
from torch.nn import functional as F
from tmrc.tmrc_core.models.components import decoder

from tmrc.tmrc_core.utils import platform

from tmrc.tmrc_core.utils.registry import register_model

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from tmrc.tmrc_core.models.components import MASK_REGISTRY


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

        if self.config.model.flex:
            #flex_attention = torch.compile(torch.nn.attention.flex_attention.flex_attention, dynamic=False)
            self.block_mask_fn = MASK_REGISTRY.get(config.model.mask)
            self.uses_flex = True
        else:
            self.uses_flex = False

    @staticmethod
    def validate_config(config):
        """Some basic sanity checks for the model config."""

        assert config.tokenizer.vocab_size is not None, "valid vocabulary size not defined"
        assert config.model.context_length is not None, "context length must be valid int > 0"
        assert config.model.d_model % config.model.n_head == 0, "d_model must be divisible by n_head"

        assert (config.model.flash and config.model.flex) is not True, "flash and flex attention cannot be used simultaneously"
        if config.model.flex:
            assert config.model.mask is not None, "Flex attention requires block mask when calling forward"

    def get_num_params(self, non_embedding=False):
        """Get total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None, doc_ids=None):
        
        B, T = idx.shape
        assert T <= self.config.model.context_length, f"Sequence length {T} > context length {self.config.model.context_length}"

        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(self.arange_T[:T]) # (B, T, C)

        x = tok_emb + pos_emb

        if self.uses_flex:
            created_block_mask = create_block_mask(self.block_mask_fn(doc_ids), \
                                               x.shape[0], None, x.shape[-1], x.shape[-1], _compile=True, device=self.platform.get_device_str())
        else:
            created_block_mask = None
        

        for block in self.transformer.h:
            x = block(x, created_block_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.loss_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss