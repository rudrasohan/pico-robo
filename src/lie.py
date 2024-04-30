import torch
from typing import Tuple

class SO3:

    def __init__(self, value:torch.Tensor):
        
        shape = value.shape
        num_dim = len(value.shape)

        if num_dim == 2:
            value = torch.reshape(value, (1, *value.shape))
        
        if value.shape[1:] != (3, 3):
            raise ValueError(f"SO3 matrices cannot be of shape {value.shape}.")
        
        if not self.is_orthonormal(value):
            raise ValueError(f"SO3 matrices should be orthonormal.")

        self.value = value
        self.num_dim = num_dim
        self.shape = shape
    
    def __repr__(self):
        return f"{self.value} shape:{self.shape}"
    
    def __str__(self):
        return f"{self.value} shape:{self.shape}"
    
    def is_orthonormal(self, value:torch.Tensor)->bool:
        vv_transpose = value @ value.transpose(1, 2)
        det = torch.det(value)
        identity = torch.eye(3)
        orthogonal = torch.allclose(vv_transpose, identity)
        det_check  = torch.allclose(det, torch.tensor([1.0]))
        return (orthogonal and det_check)
