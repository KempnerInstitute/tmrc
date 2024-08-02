from . import register_activation
import torch.nn.functional as F


@register_activation("relu")
def relu(x):
    return F.relu(x)

@register_activation("gelu")
def gelu(x):
    return F.gelu(x)