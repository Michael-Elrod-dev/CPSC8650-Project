# test_implementation.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import load_dataset, create_data_loaders
from models.resnet3d import resnet3d_18
import config

def test_data_loading():
    """Test that data loading works correctly"""
    print("Testing data loading...")
    
    # Test facial features task
    image_paths, labels = load_dataset("facial_features")
    print(f"Loaded {len(image_paths)} images for facial features task")
    print(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    # Test brain tissue loss task
    image_paths, labels = load_dataset("brain_tissue_loss")
    print(f"Loaded {len(image_paths)} images for brain tissue loss task")
    print(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    print("Data loading test completed\n")
    return len(image_paths) > 0

def test_data_loaders():
    """Test that data loaders work correctly"""
    print("Testing data loaders...")
    
    # Create data loaders with small batch size
    train_loader, val_loader, test_loader = create_data_loaders(
        "facial_features", batch_size=2, test_size=0.2, val_size=0.2
    )
    
    # Get a batch of data
    inputs, targets = next(iter(train_loader))
    
    print(f"Batch shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Visualize a slice from the first image
    plt.figure(figsize=(8, 8))
    plt.imshow(inputs[0, 0, inputs.shape[2]//2, :, :].numpy(), cmap='gray')
    plt.title(f"Target: {targets[0].item()}")
    plt.axis('off')
    plt.savefig("sample_slice.png")
    print(f"Sample slice saved to sample_slice.png")
    
    print("Data loaders test completed\n")
    return True

def test_model():
    """Test that model can be created and forward passes work"""
    print("Testing model creation and forward pass...")
    
    # Create model
    model = resnet3d_18(num_classes=1)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create random input
    batch_size = 2
    input_shape = (batch_size, 1, *config.IMAGE_SIZE)
    dummy_input = torch.randn(input_shape)
    
    # Try forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Forward pass failed with error: {str(e)}")
        return False

def test_small_training():
    """Test a small training cycle"""
    print("Testing small training cycle...")
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = resnet3d_18(num_classes=1).to(device)
    
    # Create data loaders with small batch size
    train_loader, val_loader, _ = create_data_loaders(
        "facial_features", batch_size=2, test_size=0.1, val_size=0.1
    )
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few iterations
    model.train()
    max_iterations = 5
    
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= max_iterations:
            break
            
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {i+1}, Loss: {loss.item():.4f}")
    
    print("Small training cycle completed\n")
    return True

if __name__ == "__main__":
    # Run tests
    data_loading_success = test_data_loading()
    
    if data_loading_success:
        data_loaders_success = test_data_loaders()
        model_success = test_model()
        
        if data_loaders_success and model_success:
            training_success = test_small_training()
            
            if training_success:
                print("All tests passed! The implementation appears to be working correctly.")
                print("You can now proceed with full training.")
            else:
                print("Training test failed. Debug the training process.")
        else:
            print("Data loaders or model test failed. Fix these issues before proceeding.")
    else:
        print("Data loading failed. Check your file paths and data structure.")