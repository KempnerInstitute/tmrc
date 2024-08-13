import torch

import torch.nn as nn
from torch.nn import functional as F

from . import ACTIVATION_REGISTRY

class CausalSelfAttention(nn.Module):
    """
    A simple adapation of vanilla self attention head with a causal mask.
    Minor changes from Karpathy's GPT implementation.
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads==0
        self.c_attn = nn.Linear(config.d_model, 3*config.d_model, bias=config.attn_bias) # easier to do K,V,Q at once
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.attn_dropout_p)
        self.proj_dropout = nn.Dropout(config.proj_dropout_p)
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("Warning: flash attention not found (torch >= 2.0)")
            self.register_buffer("causal_mask",
                                 torch.tril(torch.ones(config.ctx_len, config.ctx_len)).view(1, 1, config.ctx_len, config.ctx_len))

    def forward(self, x):
        # x: (B, T, C)

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2) # (B, n_h, T, d_head)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)

        if self.flash:
            y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            w = (q @ k.transpose(-2, -1))*k.size(-1)**(-0.5)
            w = w.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            w = F.softmax(w, dim=-1)
            w = self.attn_dropout(w)
            y = w @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, config.mlp_scale_factor * config.d_model, bias=config.mlp_bias)
        self.activation = ACTIVATION_REGISTRY.get(config.activation)
        self.c_proj  = nn.Linear(config.mlp_scale_factor * config.d_model, config.d_model, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.mlp_dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x