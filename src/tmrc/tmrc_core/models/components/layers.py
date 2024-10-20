from tmrc.tmrc_core.utils.registry import register_activation, register_mask
import torch.nn.functional as F
import torch


"""Activation functions"""
@register_activation("relu")
def relu(x):
    return F.relu(x)

@register_activation("gelu")
def gelu(x):
    return F.gelu(x)


"""Masks"""
@register_mask("causal")
def causal_mask(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))

@register_mask("causal_document")
def get_doc_causal_mask_fn(doc_mask_ids):
    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_mask_ids[b, q_idx] == doc_mask_ids[b,kv_idx]
        return causal_mask & document_mask
    return document_causal_mask