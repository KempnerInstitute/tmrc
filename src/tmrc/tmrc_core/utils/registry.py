class Registry:
    def __init__(self, name):
        self.name = name
        self._registry = {}
    
    def register(self, key=None):
        def decorator(obj):
            k = key or obj.__name__
            if k in self._registry:
                raise ValueError(f"{k} already registered in {self.name}")
            self._registry[k] = obj
            return obj
        
        return decorator
    
    def get(self, key):
        if key not in self._registry:
            raise KeyError(f"{key} not found in {self.name}")
        return self._registry[key]
    
    def list_keys(self):
        return list(self._registry.keys())
    
"""Activation registry"""
ACTIVATION_REGISTRY = Registry("activations")
register_activation = ACTIVATION_REGISTRY.register

"""Optimizer registry"""
OPTIMIZER_REGISTRY = Registry("optimizers")
register_optimizer = OPTIMIZER_REGISTRY.register

"""Model registry"""
MODEL_REGISTRY = Registry("models")
register_model = MODEL_REGISTRY.register