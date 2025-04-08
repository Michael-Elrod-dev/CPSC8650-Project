# utils/train_utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from collections import defaultdict
import config

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        device (str): Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
    
    # Calculate epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """
    Validate model
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (str): Device to validate on
        
    Returns:
        tuple: (average loss, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate validation loss
    val_loss = running_loss / len(dataloader.dataset)
    
    return val_loss, np.array(all_preds), np.array(all_targets)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=None, scheduler=None, early_stopping_patience=None, 
                model_save_path=None):
    """
    Train and validate model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        device (str): Device to train on
        num_epochs (int, optional): Number of epochs to train
        scheduler (LRScheduler, optional): Learning rate scheduler
        early_stopping_patience (int, optional): Patience for early stopping
        model_save_path (str, optional): Path to save the best model
        
    Returns:
        tuple: (trained model, history dictionary)
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if early_stopping_patience is None:
        early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    if model_save_path is None:
        model_save_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    
    # Initialize variables
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    
    return model, history

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history (dict): Training history
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate if available
    if 'lr' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def k_fold_cross_validation(model_fn, fold_loaders, criterion, optimizer_fn, device,
                           num_epochs=None, scheduler_fn=None, early_stopping_patience=None):
    """
    Perform k-fold cross-validation
    
    Args:
        model_fn (callable): Function to create a new model instance
        fold_loaders (list): List of (train_loader, val_loader) tuples for each fold
        criterion (nn.Module): Loss function
        optimizer_fn (callable): Function to create a new optimizer instance
        device (str): Device to train on
        num_epochs (int, optional): Number of epochs to train
        scheduler_fn (callable, optional): Function to create a new scheduler instance
        early_stopping_patience (int, optional): Patience for early stopping
        
    Returns:
        tuple: (list of trained models, list of histories, average validation loss)
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if early_stopping_patience is None:
        early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    
    models = []
    histories = []
    val_losses = []
    
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"Fold {fold+1}/{len(fold_loaders)}")
        
        # Create new model and optimizer for each fold
        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        
        # Create scheduler if provided
        scheduler = None
        if scheduler_fn is not None:
            scheduler = scheduler_fn(optimizer)
        
        # Train model for this fold
        model_save_path = os.path.join(config.MODEL_DIR, f"best_model_fold_{fold+1}.pth")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=num_epochs, scheduler=scheduler,
            early_stopping_patience=early_stopping_patience,
            model_save_path=model_save_path
        )
        
        # Store results
        models.append(model)
        histories.append(history)
        val_losses.append(min(history['val_loss']))
        
        print(f"Fold {fold+1} completed. Best validation loss: {val_losses[-1]:.4f}")
        print("-" * 50)
    
    # Calculate average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Cross-validation completed. Average validation loss: {avg_val_loss:.4f}")
    
    return models, histories, avg_val_loss