# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from datetime import datetime

import config
from data.dataset import create_data_loaders, create_kfold_loaders
from data.augmentation import get_transform
from models.resnet3d import resnet3d_18, resnet3d_34
from utils.train_utils import train_model, k_fold_cross_validation, plot_training_history
from utils.eval_utils import evaluate_model, calculate_metrics

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data transformations
    transform = get_transform(mode='train')
    
    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.OUTPUT_DIR, f"{args.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate class weights for balanced loss function
    if args.task == "facial_features":
        # Get training data labels
        _, train_labels = load_dataset(args.task)
        # Count positive and negative samples
        pos_count = sum(train_labels)
        neg_count = len(train_labels) - pos_count
        # Calculate weights inversely proportional to class frequencies
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        class_weight = torch.tensor([pos_weight], device=device)
        print(f"Using class weight: {pos_weight:.4f} for positive class")
    else:
        class_weight = None
    
    # Create model
    if args.model == 'resnet18':
        model_fn = lambda: resnet3d_18(num_classes=1, dropout_rate=0.3)
    elif args.model == 'resnet34':
        model_fn = lambda: resnet3d_34(num_classes=1, dropout_rate=0.3)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Define loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    
    # Define optimizer function with weight decay for L2 regularization
    optimizer_fn = lambda params: optim.Adam(params, lr=config.LEARNING_RATE, weight_decay=0.001)
    
    # Define scheduler function
    scheduler_fn = lambda optimizer: ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    if args.cross_val:
        # K-fold cross-validation
        print(f"Performing {config.NUM_FOLDS}-fold cross-validation for task: {args.task}")
        
        # Create k-fold data loaders
        fold_loaders = create_kfold_loaders(
            args.task, transform=transform, batch_size=args.batch_size,
            n_folds=config.NUM_FOLDS, seed=config.RANDOM_SEED
        )
        
        # Perform k-fold cross-validation
        models, histories, avg_val_loss = k_fold_cross_validation(
            model_fn, fold_loaders, criterion, optimizer_fn, device,
            num_epochs=args.epochs, scheduler_fn=scheduler_fn,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )
        
        # Plot history for each fold
        for i, history in enumerate(histories):
            plot_path = os.path.join(output_dir, f"fold_{i+1}_history.png")
            plot_training_history(history, save_path=plot_path)
        
        # Save results summary
        with open(os.path.join(output_dir, "cross_val_results.txt"), "w") as f:
            f.write(f"Task: {args.task}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Number of folds: {config.NUM_FOLDS}\n")
            f.write(f"Average validation loss: {avg_val_loss:.4f}\n")
            f.write("\nFold results:\n")
            for i, history in enumerate(histories):
                f.write(f"Fold {i+1} best validation loss: {min(history['val_loss']):.4f}\n")
    else:
        # Single train/validation/test run
        print(f"Training model for task: {args.task}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            args.task, transform=transform, batch_size=args.batch_size,
            test_size=config.TRAIN_VAL_TEST_SPLIT[2], 
            val_size=config.TRAIN_VAL_TEST_SPLIT[1]/(1-config.TRAIN_VAL_TEST_SPLIT[2]),
            seed=config.RANDOM_SEED
        )
        
        # Create model
        model = model_fn().to(device)
        
        # Create optimizer
        optimizer = optimizer_fn(model.parameters())
        
        # Create scheduler
        scheduler = scheduler_fn(optimizer)
        
        # Train model
        model_save_path = os.path.join(output_dir, "best_model.pth")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, scheduler=scheduler,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            model_save_path=model_save_path
        )
        
        # Plot training history
        plot_path = os.path.join(output_dir, "training_history.png")
        plot_training_history(history, save_path=plot_path)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)
        
        # Calculate metrics
        metrics = calculate_metrics(test_targets, test_preds)
        
        # Save test results
        with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
            f.write(f"Task: {args.task}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Test loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Sensitivity: {metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"F1 score: {metrics['f1']:.4f}\n")
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
        
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D ResNet for brain MRI analysis")
    parser.add_argument("--task", type=str, choices=config.TASKS, required=True,
                        help="Task to train on")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet34"], default="resnet18",
                        help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Number of epochs to train")
    parser.add_argument("--cross_val", action="store_true",
                        help="Perform cross-validation")
    
    args = parser.parse_args()
    main(args)