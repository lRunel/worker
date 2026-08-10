"""
Trainer — Model-agnostic local training engine for workers.

Receives a model name, config, and state dict from the controller,
reconstructs the model, runs local training steps, and returns gradients.
"""
import torch
import torch.nn as nn
from models import get_model


class Trainer:
    def __init__(self, model_name: str, model_config: dict, model_state_dict: dict, lr: float = 0.001):
        self.model = get_model(model_name, model_config)
        self.model.load_state_dict(model_state_dict)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def train_steps(self, data: torch.Tensor, labels: torch.Tensor, steps: int):
        self.model.train()
        batch_size = 64  # must match controller
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Shuffle received data for better gradient diversity
        perm = torch.randperm(len(data))
        data   = data[perm]
        labels = labels[perm]

        # Snapshot weights before local training
        old_state = [p.clone().detach() for p in self.model.parameters()]

        n_samples = len(data)
        for step in range(steps):
            # Cycle through data (enables multi-epoch local training)
            idx = (step * batch_size) % n_samples
            end = idx + batch_size
            if end <= n_samples:
                batch_data   = data[idx:end]
                batch_labels = labels[idx:end]
            else:
                # Wrap around
                batch_data   = torch.cat([data[idx:], data[:end - n_samples]])
                batch_labels = torch.cat([labels[idx:], labels[:end - n_samples]])

            optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss    = self.criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Compute pseudo-gradients: old_weights - new_weights
        # The controller applies this via SGD(lr=1.0) -> params -= 1.0 * (old - new) -> new
        pseudo_grads = []
        for old_p, new_p in zip(old_state, self.model.parameters()):
            pseudo_grads.append(old_p - new_p.detach())

        return pseudo_grads
