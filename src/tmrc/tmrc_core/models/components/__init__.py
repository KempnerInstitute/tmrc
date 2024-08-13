from ...utils.registry import Registry

ACTIVATION_REGISTRY = Registry("activations")
register_activation = ACTIVATION_REGISTRY.register