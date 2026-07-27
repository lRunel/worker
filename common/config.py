# config.py — shared configuration for controller and workers
# Supports YAML-based config loading with sensible defaults.

import yaml

# Networking
CONTROLLER_HOST = "0.0.0.0"
CONTROLLER_PORT = 8000

def get_controller_url(ip: str = "127.0.0.1", port: int = 8000) -> str:
    return f"http://{ip}:{port}"

import torch
DEVICE = "cpu"  # Workers may be CPU-only

# ── Default training hyperparameters (overridden by YAML config) ──
LEARNING_RATE = 0.001
MAX_STALENESS = 5
STEPS_PER_TASK = 15
BATCH_SIZE = 64
TARGET_EPOCHS = 5
TRAIN_DATASET_SIZE = 60000  # MNIST training set size


def load_config(yaml_path: str) -> dict:
    """Load a YAML training config file and return the parsed dict.
    
    Example config structure:
        model:
          name: cnn
          params: {in_channels: 1, num_classes: 10, input_size: 28}
        dataset:
          name: mnist
          data_dir: ./data
        training:
          learning_rate: 0.001
          batch_size: 64
          target_epochs: 5
          max_staleness: 5
          steps_per_task: 15
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Merge training section into module-level defaults for backward compat
    training = config.get("training", {})
    defaults = {
        "learning_rate": LEARNING_RATE,
        "max_staleness": MAX_STALENESS,
        "steps_per_task": STEPS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "target_epochs": TARGET_EPOCHS,
    }
    for key, default_val in defaults.items():
        training.setdefault(key, default_val)
    config["training"] = training

    # Ensure model and dataset sections exist
    config.setdefault("model", {"name": "cnn", "params": {}})
    config.setdefault("dataset", {"name": "mnist", "data_dir": "./data"})

    return config
