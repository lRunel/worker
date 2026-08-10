"""
Model Registry — maps architecture names to model classes.

Usage:
    from models import get_model, list_models
    model = get_model("cnn", {"in_channels": 1, "num_classes": 10, "input_size": 28})
"""
import torch.nn as nn

# Registry dict: name -> (class, description)
_REGISTRY: dict[str, tuple[type[nn.Module], str]] = {}


def register(name: str, description: str = ""):
    """Decorator to register a model class under a given name."""
    def decorator(cls: type[nn.Module]):
        _REGISTRY[name] = (cls, description)
        return cls
    return decorator


def get_model(name: str, params: dict | None = None) -> nn.Module:
    """Instantiate a registered model by name with the given config params."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    cls, _ = _REGISTRY[name]
    return cls(**(params or {}))


def list_models() -> list[dict]:
    """Return a list of all registered models with their names and descriptions."""
    return [
        {"name": name, "description": desc}
        for name, (_, desc) in _REGISTRY.items()
    ]


# Import all architecture modules so they self-register on import
from . import cnn, mlp, rnn, transformer, neural_net, cifar_cnn  # noqa: F401, E402
