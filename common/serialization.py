import io
import base64
import torch

def encode_tensor(t: torch.Tensor) -> str:
    """Serializes a PyTorch tensor to base64 encoded string."""
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_tensor(s: str) -> torch.Tensor:
    """Deserializes a base64 encoded string to PyTorch tensor."""
    b = base64.b64decode(s)
    buffer = io.BytesIO(b)
    return torch.load(buffer, weights_only=True)

def encode_state_dict(state_dict: dict) -> str:
    """Serializes a model state dictionary to base64 encoded string."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_state_dict(s: str) -> dict:
    """Deserializes a base64 encoded string to model state dictionary."""
    b = base64.b64decode(s)
    buffer = io.BytesIO(b)
    return torch.load(buffer, weights_only=True)

def encode_gradients(grads: list[torch.Tensor]) -> str:
    """Serializes a list of gradient tensors to base64 encoded string."""
    buffer = io.BytesIO()
    torch.save(grads, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_gradients(s: str) -> list[torch.Tensor]:
    """Deserializes a base64 encoded string to a list of gradient tensors."""
    b = base64.b64decode(s)
    buffer = io.BytesIO(b)
    return torch.load(buffer, weights_only=True)

def sparsify_with_error_feedback(
    grads: list[torch.Tensor],
    residuals: list[torch.Tensor] | None,
    sparsity_ratio: float,
) -> tuple[list[dict], list[torch.Tensor]]:
    """
    Compresses gradients using Top-K sparsification with Error Feedback (EF-SGD).
    Accumulates untransmitted gradient coordinates into residuals so no gradient information is lost.

    Parameters
    ----------
    grads          : list of dense gradient tensors
    residuals      : list of accumulated residual tensors (or None on first round)
    sparsity_ratio : fraction of coordinates to retain [0.001, 1.0]

    Returns
    -------
    sparse_data   : list of dicts for serialization
    new_residuals : list of updated residual tensors to keep locally for the next round
    """
    ratio = max(0.001, min(1.0, float(sparsity_ratio)))

    if residuals is None or len(residuals) != len(grads):
        residuals = [torch.zeros_like(g) for g in grads]

    if ratio >= 0.99:
        sparse_data = [{"shape": list(g.shape), "dense": g + r.to(g.device)} for g, r in zip(grads, residuals)]
        new_residuals = [torch.zeros_like(g) for g in grads]
        return sparse_data, new_residuals

    sparse_data = []
    new_residuals = []

    for g, r in zip(grads, residuals):
        # Accumulate residual into current gradient
        g_acc = g + r.to(g.device)
        flat_g = g_acc.flatten()
        n_elements = flat_g.numel()
        k = max(1, int(n_elements * ratio))

        if k < n_elements:
            abs_g = torch.abs(flat_g)
            _, indices = torch.topk(abs_g, k)
            actual_values = flat_g[indices]

            sparse_data.append({
                "shape": list(g.shape),
                "indices": indices.to(torch.int32),
                "values": actual_values,
            })

            # New residual: untransmitted coordinates remain in local memory
            res_flat = flat_g.clone()
            res_flat[indices] = 0.0
            new_residuals.append(res_flat.view(g.shape).detach())
        else:
            sparse_data.append({
                "shape": list(g.shape),
                "dense": g_acc,
            })
            new_residuals.append(torch.zeros_like(g))

    return sparse_data, new_residuals


def sparsify_gradients(grads: list[torch.Tensor], sparsity_ratio: float) -> list[dict]:
    """
    Compresses gradients using Top-K sparsification based on the target sparsity ratio.
    Returns a list of dictionaries containing {shape, indices, values}.
    """
    sparse_data, _ = sparsify_with_error_feedback(grads, None, sparsity_ratio)
    return sparse_data

def desparsify_gradients(sparse_data: list[dict], device: str = "cpu") -> list[torch.Tensor]:
    """
    Reconstructs dense gradient tensors from sparse dictionaries.
    """
    grads = []
    for data in sparse_data:
        shape = data["shape"]
        if "dense" in data:
            grads.append(data["dense"].to(device))
        else:
            indices = data["indices"].to(torch.int64).to(device)
            values = data["values"].to(device)
            flat_g = torch.zeros((torch.prod(torch.tensor(shape)).item(),), device=device, dtype=values.dtype)
            flat_g[indices] = values
            grads.append(flat_g.view(shape))
    return grads

def encode_sparse_gradients(sparse_grads: list[dict]) -> str:
    """Serializes sparse gradient dictionaries."""
    buffer = io.BytesIO()
    torch.save(sparse_grads, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_sparse_gradients(s: str) -> list[dict]:
    """Deserializes to sparse gradient dictionaries."""
    b = base64.b64decode(s)
    buffer = io.BytesIO(b)
    return torch.load(buffer, weights_only=False) # Needs to load dicts and tensors
