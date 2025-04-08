# evaluate.py

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt

import config
from data.dataset import create_data_loaders
from models.resnet3d import resnet3d_18, resnet3d_34
from utils.eval_utils import (
    evaluate_model, calculate_metrics, plot_roc_curve, 
    plot_precision_recall_curve, plot_confusion_matrix,
    compare_models, paired_t_test
)

def main(args):
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(config.OUTPUT_DIR, f"evaluation_{args.task}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders (no augmentation for evaluation)
    _, _, test_loader = create_data_loaders(
        args.task, transform=None, batch_size=args.batch_size,
        test_size=0.2, val_size=0.125, seed=config.RANDOM_SEED
    )
    
    # Load model
    if args.model == 'resnet18':
        model = resnet3d_18(num_classes=1)
    elif args.model == 'resnet34':
        model = resnet3d_34(num_classes=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)
    
    # Calculate metrics
    metrics = calculate_metrics(test_targets, test_preds)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
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
    
    # Plot ROC curve
    plot_roc_curve(metrics, save_path=os.path.join(output_dir, "roc_curve.png"))
    
    # Plot precision-recall curve
    plot_precision_recall_curve(metrics, save_path=os.path.join(output_dir, "pr_curve.png"))
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    
    # Compare with reference model if reference metrics are provided
    if args.reference_file:
        # Load reference metrics
        reference_metrics = {}
        with open(args.reference_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    try:
                        value = float(value.strip())
                        reference_metrics[key] = value
                    except ValueError:
                        pass
        
        # Compare with reference model
        comparison_table, is_better = compare_models(reference_metrics, metrics, args.task)
        
        # Print comparison
        print("\nComparison with reference model:")
        for i in range(len(comparison_table['Metric'])):
            print(f"{comparison_table['Metric'][i]}: {comparison_table['Reference'][i]:.4f} (Reference) vs {comparison_table['Our Model'][i]:.4f} (Our Model)")
        
        # Save comparison
        with open(os.path.join(output_dir, "comparison.txt"), "w") as f:
            f.write("Comparison with reference model:\n")
            for i in range(len(comparison_table['Metric'])):
                f.write(f"{comparison_table['Metric'][i]}: {comparison_table['Reference'][i]:.4f} (Reference) vs {comparison_table['Our Model'][i]:.4f} (Our Model)\n")
            f.write(f"\nOverall, our model is {'better' if is_better else 'not better'} than the reference model.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D ResNet for brain MRI analysis")
    parser.add_argument("--task", type=str, choices=config.TASKS, required=True,
                        help="Task to evaluate on")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet34"], default="resnet18",
                        help="Model architecture used")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for evaluation")
    parser.add_argument("--reference_file", type=str,
                        help="Path to file containing reference model metrics")
    
    args = parser.parse_args()
    main(args)