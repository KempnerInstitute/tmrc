import torch

import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    Super basic implementation of a causal self-attention head.
    """