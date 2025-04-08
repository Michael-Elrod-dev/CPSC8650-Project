# data/augmentation.py

import torch
import torch.nn.functional as F
import kornia
import random

class Compose:
    """Compose several transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class RandomRotate3D:
    """Randomly rotate the 3D volume using kornia"""
    def __init__(self, max_angle=10):
        self.max_angle = max_angle
        
    def __call__(self, x):
        # Generate random angles
        angles = torch.rand(3) * 2 * self.max_angle - self.max_angle
        
        # Add batch dimension
        x_batch = x.unsqueeze(0)
        
        # Apply rotation
        rotated = kornia.geometry.transform.rotate3d(
            x_batch,
            angles[0], angles[1], angles[2],
            center=None,
            mode='bilinear',
            padding_mode='zeros'
        )
        
        # Remove batch dimension
        return rotated.squeeze(0)

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