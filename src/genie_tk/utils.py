'''
Utility Functions for Genie-TK

License: MIT
'''

from typing import Callable, Dict, Any, Tuple
import torch
from torch import Tensor


def has_cuda_kernels() -> bool:
    #Check if CUDA kernels are available.#
    try:
        from genie_tk import _C
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_info() -> Dict[str, Any]:
    #Get information about available devices.#
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_kernels_available": has_cuda_kernels(),
        "device_count": 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": props.name,
                "total_memory": props.total_memory,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })
    
    return info


def permute_final_dims(tensor: Tensor, *dims: int) -> Tensor:
    #
    Permute the last dimensions of a tensor.
    
    Args:
        tensor: Input tensor
        *dims: Dimension indices (relative to the last dimensions)
        
    Returns:
        Permuted tensor
    #
    num_dims = len(dims)
    # Build full permutation
    rank = tensor.dim()
    perm = list(range(rank - num_dims)) + [rank - num_dims + d for d in dims]
    return tensor.permute(perm)


def flatten_final_dims(tensor: Tensor, num_dims: int) -> Tensor:
    #
    Flatten the last num_dims dimensions.
    
    Args:
        tensor: Input tensor
        num_dims: Number of trailing dimensions to flatten
        
    Returns:
        Flattened tensor
    #
    shape = list(tensor.shape[:-num_dims]) + [-1]
    return tensor.reshape(shape)


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Tensor],
    chunk_size: int,
    no_batch_dims: int,
) -> Tensor:
    #
    Process a layer in chunks to reduce memory usage.
    
    Args:
        layer: Callable layer to apply
        inputs: Dictionary of input tensors
        chunk_size: Size of each chunk
        no_batch_dims: Number of batch dimensions
        
    Returns:
        Output tensor
    #
    # Get first input to determine shape
    first_key = next(iter(inputs))
    first_tensor = inputs[first_key]
    
    # Flatten batch dimensions
    orig_batch_shape = first_tensor.shape[:no_batch_dims]
    batch_size = 1
    for d in orig_batch_shape:
        batch_size *= d
    
    # Reshape inputs
    flat_inputs = {}
    for key, tensor in inputs.items():
        flat_inputs[key] = tensor.reshape(batch_size, *tensor.shape[no_batch_dims:])
    
    # Process in chunks
    outputs = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk_inputs = {k: v[start:end] for k, v in flat_inputs.items()}
        chunk_output = layer(**chunk_inputs)
        outputs.append(chunk_output)
    
    # Concatenate and reshape
    output = torch.cat(outputs, dim=0)
    output = output.reshape(*orig_batch_shape, *output.shape[1:])
    
    return output
