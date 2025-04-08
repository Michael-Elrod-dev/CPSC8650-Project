# data/dataset.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import nibabel as nib
import config

class BrainMRIDataset(Dataset):
    """
    Dataset class for loading and preprocessing brain MRI data
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load 3D MRI volume
        img_path = self.image_paths[idx]
        # Nibabel is used for medical image formats
        img = nib.load(img_path).get_fdata()
        
        # Resize to fixed dimensions if needed
        if img.shape != config.IMAGE_SIZE:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors for each dimension
            factors = [t / s for t, s in zip(config.IMAGE_SIZE, img.shape)]
            img = zoom(img, factors, order=1)  # order=1 for linear interpolation
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Add channel dimension
        
        # Apply transformations
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return img_tensor, label

def load_dataset(task, data_dir=None):
    """
    Load and preprocess the dataset
    
    Args:
        task (str): 'facial_features' or 'brain_tissue_loss'
        data_dir (str): Directory containing the dataset
        
    Returns:
        tuple: (image_paths, labels)
    """
    if data_dir is None:
        data_dir = os.path.join(config.DATA_DIR, "raw")
    
    # Path to the files
    image_dir = os.path.join(data_dir, "files")
    
    # Path to the labels file
    label_file = os.path.join(data_dir, "labels.csv")
    
    # Read the labels CSV file
    df = pd.read_csv(label_file)
    
    # Process image paths and labels
    image_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        filename = row['Filename']
        img_path = os.path.join(image_dir, filename)
        
        # Check if the file exists
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue
        
        # Get the appropriate label based on the task
        if task == "facial_features":
            # Convert 'Yes'/'No' to 1/0
            label = 1 if row['Recognizable-Facial-Feature'] == 'Yes' else 0
        elif task == "brain_tissue_loss":
            label = 1 if row['Brain-Feature-Loss'] == 'Yes' else 0
        else:
            raise ValueError(f"Unknown task: {task}")
        
        image_paths.append(img_path)
        labels.append(label)
    
    return image_paths, labels

def create_data_loaders(task, transform=None, batch_size=None, test_size=None, val_size=None, seed=None):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        task (str): 'facial_features' or 'brain_tissue_loss'
        transform (callable, optional): Transformations to apply to the data
        batch_size (int, optional): Batch size
        test_size (float, optional): Proportion of data to use for testing
        val_size (float, optional): Proportion of training data to use for validation
        seed (int, optional): Random seed
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if test_size is None:
        test_size = config.TRAIN_VAL_TEST_SPLIT[2]
    if val_size is None:
        val_size = config.TRAIN_VAL_TEST_SPLIT[1] / (1 - config.TRAIN_VAL_TEST_SPLIT[2])
    if seed is None:
        seed = config.RANDOM_SEED
    
    # Load dataset
    image_paths, labels = load_dataset(task)
    
    if len(image_paths) == 0:
        raise ValueError("No valid images found in the dataset")
    
    print(f"Dataset loaded: {len(image_paths)} images for {task} task")
    
    # Split data into train+val and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    
    # Split train+val into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size, random_state=seed, stratify=train_val_labels
    )
    
    # Create datasets
    train_dataset = BrainMRIDataset(train_paths, train_labels, transform=transform)
    val_dataset = BrainMRIDataset(val_paths, val_labels, transform=None)  # No augmentation for validation
    test_dataset = BrainMRIDataset(test_paths, test_labels, transform=None)  # No augmentation for testing
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data split: {len(train_paths)} training, {len(val_paths)} validation, {len(test_paths)} testing")
    
    return train_loader, val_loader, test_loader

def create_kfold_loaders(task, transform=None, batch_size=None, n_folds=None, seed=None):
    """
    Create data loaders for k-fold cross-validation
    
    Args:
        task (str): 'facial_features' or 'brain_tissue_loss'
        transform (callable, optional): Transformations to apply to the data
        batch_size (int, optional): Batch size
        n_folds (int, optional): Number of folds
        seed (int, optional): Random seed
        
    Returns:
        list: List of (train_loader, val_loader) tuples for each fold
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if n_folds is None:
        n_folds = config.NUM_FOLDS
    if seed is None:
        seed = config.RANDOM_SEED
    
    # Load dataset
    image_paths, labels = load_dataset(task)
    
    if len(image_paths) == 0:
        raise ValueError("No valid images found in the dataset")
    
    print(f"Dataset loaded: {len(image_paths)} images for {task} task")
    
    # Create k-fold cross-validation splits
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Convert labels to numpy array
    labels_array = np.array(labels)
    
    # Create a list to store data loaders for each fold
    fold_loaders = []
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(image_paths)):
        # Split data based on fold indices
        train_paths = [image_paths[i] for i in train_indices]
        train_labels = labels_array[train_indices].tolist()
        val_paths = [image_paths[i] for i in val_indices]
        val_labels = labels_array[val_indices].tolist()
        
        # Create datasets
        train_dataset = BrainMRIDataset(train_paths, train_labels, transform=transform)
        val_dataset = BrainMRIDataset(val_paths, val_labels, transform=None)  # No augmentation for validation
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
        
        print(f"Fold {fold+1}: {len(train_paths)} training, {len(val_paths)} validation")
    
    return fold_loaders