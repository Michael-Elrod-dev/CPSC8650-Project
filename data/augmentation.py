# data/augmentation.py

import torch
import torch.nn.functional as F
import numpy as np
import random

class Compose:
    """Compose several transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class RandomRotate:
    """Randomly rotate the 3D volume"""
    def __init__(self, max_angle=10):
        self.max_angle = max_angle
        
    def __call__(self, x):
        # For simplicity, we'll only implement rotation around z-axis
        # In a real implementation, you might want to rotate around all axes
        angle = random.uniform(-self.max_angle, self.max_angle) * (np.pi / 180)
        
        # Create rotation matrix
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        rot_matrix = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0],
            [0, 0, 1]
        ], dtype=torch.float)
        
        # Apply rotation
        # In a real implementation, you would use proper 3D rotation
        # Here we're just illustrating the concept
        # For a proper implementation, consider using libraries like kornia
        
        # For simplicity, we'll just return the original tensor
        # as proper 3D rotation is complex to implement from scratch
        return x

class RandomFlip:
    """Randomly flip the 3D volume along specified axes"""
    def __init__(self, axes=(0, 1, 2)):
        self.axes = axes
        
    def __call__(self, x):
        # Choose a random axis to flip
        axis = random.choice(self.axes)
        
        # Flip along the chosen axis (adjust for channel dimension)
        # Note: x has shape [C, D, H, W] where C is the channel dimension
        return torch.flip(x, dims=[axis+1])

class RandomGaussianNoise:
    """Add random Gaussian noise to the 3D volume"""
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

class RandomIntensityShift:
    """Randomly shift intensity values"""
    def __init__(self, shift_range=(-0.1, 0.1)):
        self.shift_range = shift_range
        
    def __call__(self, x):
        shift = random.uniform(*self.shift_range)
        return x + shift

def get_transform(mode='train'):
    """
    Get transformation pipeline based on mode
    
    Args:
        mode (str): 'train' or 'test'
        
    Returns:
        callable: Transformation function
    """
    if mode == 'train':
        return Compose([
            RandomFlip(),
            RandomGaussianNoise(),
            RandomIntensityShift()
        ])
    else:
        # No augmentation for test/validation
        return None