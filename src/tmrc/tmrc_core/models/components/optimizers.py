from tmrc.tmrc_core.utils.registry import register_optimizer
from typing import Iterable, Optional

import torch
from torch.optim import AdamW as TorchAdamW

@register_optimizer("AdamW")
def AdamW(params: Iterable[torch.nn.parameter.Parameter],
          lr: float = 1e-3,
          betas: tuple[float, float] = (0.9, 0.999),
          eps: float = 1e-8,
          weight_decay: float = 1e-2,
          amsgrad: bool = False,
          maximize: bool = False,
          foreach: Optional[bool] = None,
          capturable: bool = False,
          differentiable: bool = False,
          fused: Optional[bool] = None):
    return TorchAdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                      amsgrad=amsgrad, maximize=maximize, foreach=foreach,
                      capturable=capturable, differentiable=differentiable, fused=fused)