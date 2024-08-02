from ..utils.registry import Registry

MODEL_REGISTRY = Registry("models")
register_model = MODEL_REGISTRY.register