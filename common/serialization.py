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

def sparsify_gradients(grads: list[torch.Tensor], sparsity_ratio: float) -> list[dict]:
    """
    Compresses gradients using Top-K sparsification based on the target sparsity ratio.
    Returns a list of dictionaries containing {shape, indices, values}.
    """
    # Clamp sparsity ratio to valid range (1.0 means 100% kept, 0.01 means 1% kept)
    ratio = max(0.001, min(1.0, float(sparsity_ratio)))
    
    if ratio >= 0.99:
        # Fast path: no sparsification needed
        return [{"shape": list(g.shape), "dense": g} for g in grads]

    sparse_data = []
    for g in grads:
        flat_g = g.flatten()
        n_elements = flat_g.numel()
        k = max(1, int(n_elements * ratio))
        
        # Get top-k absolute values
        abs_g = torch.abs(flat_g)
        if k < n_elements:
            # torch.topk is faster than sorting
            values, indices = torch.topk(abs_g, k)
            actual_values = flat_g[indices]
            sparse_data.append({
                "shape": list(g.shape),
                "indices": indices.to(torch.int32),
                "values": actual_values
            })
        else:
            # Fallback to dense if k is large
            sparse_data.append({
                "shape": list(g.shape),
                "dense": g
            })
            
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
