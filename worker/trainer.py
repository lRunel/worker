"""
Trainer — Model-agnostic local training engine for workers.

Receives a model name, config, and state dict from the controller,
reconstructs the model, runs local training steps, and returns gradients.
"""
import torch
import torch.nn as nn
from models import get_model


class Trainer:
    def __init__(self, model_name: str, model_config: dict, model_state_dict: dict):
        self.model = get_model(model_name, model_config)
        self.model.load_state_dict(model_state_dict)
        self.criterion = nn.CrossEntropyLoss()

    def train_steps(self, data: torch.Tensor, labels: torch.Tensor, steps: int):
        self.model.train()
        batch_size = 64  # must match controller

        # Shuffle received data for better gradient diversity
        perm = torch.randperm(len(data))
        data   = data[perm]
        labels = labels[perm]

        self.model.zero_grad()
        valid_steps = 0

        for step in range(steps):
            start = step * batch_size
            end   = start + batch_size
            if start >= len(data):
                break
            batch_data   = data[start:end]
            batch_labels = labels[start:end]
            if len(batch_data) == 0:
                break

            outputs = self.model(batch_data)
            loss    = self.criterion(outputs, batch_labels)
            loss.backward()  # accumulate gradients
            valid_steps += 1

        # Normalise accumulated gradients by actual number of steps run
        if valid_steps > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= valid_steps

        grads = []
        for param in self.model.parameters():
            grads.append(param.grad.clone() if param.grad is not None else torch.zeros_like(param))

        return grads
