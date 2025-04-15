# Brain MRI Quality Evaluation Project

## Project Purpose

This project implements deep learning methods to evaluate the quality of skull-stripped brain MRI images. It focuses on two critical aspects of quality assessment:

1. **Facial Feature Detection**: Identifying whether any recognizable facial features remain in skull-stripped images, which could pose privacy concerns.

2. **Brain Tissue Loss Detection**: Determining if any brain tissue voxels were accidentally removed during the skull-stripping process, which would compromise data integrity.

The project implements 3D ResNet architectures (ResNet-18 and ResNet-34) as alternatives to the multi-kernel 3D CNN with inception module presented in the reference paper, and compares their performance on both tasks.

## File Structure

```
brain_mri_analysis/
├── data/
│   ├── raw/
│   │   ├── files/          # Contains the extracted .nii files
│   │   │   ├── extracted/  # Directory with extracted NIfTI files
│   │   │   ├── *.nii.gz    # Original compressed NIfTI files
│   │   └── labels.csv      # Labels for both tasks (facial features and brain tissue loss)
│   ├── __init__.py
│   ├── dataset.py          # Dataset loading and preprocessing
│   └── augmentation.py     # Data augmentation functions
├── models/
│   ├── __init__.py
│   └── resnet3d.py         # 3D ResNet implementation
├── utils/
│   ├── __init__.py
│   ├── train_utils.py      # Training functions
│   └── eval_utils.py       # Evaluation metrics and functions
├── output/                 # Training outputs and results
│   └── models/             # Saved model weights
├── config.py               # Configuration parameters
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── requirements.txt        # Dependencies
```

## Setup and Installation

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The data has been extracted from the original .nii.gz files to the `data/raw/files/extracted` directory.

## Usage

### Training

You can run the following commands to train models for both tasks with both architectures:

1. Train ResNet-18 for facial feature detection:
   ```bash
   python train.py --task facial_features --model resnet18 --batch_size 10 --epochs 50
   ```

2. Train ResNet-34 for facial feature detection:
   ```bash
   python train.py --task facial_features --model resnet34 --batch_size 10 --epochs 50
   ```

3. Train ResNet-18 for brain tissue loss detection:
   ```bash
   python train.py --task brain_tissue_loss --model resnet18 --batch_size 10 --epochs 50
   ```

4. Train ResNet-34 for brain tissue loss detection:
   ```bash
   python train.py --task brain_tissue_loss --model resnet34 --batch_size 10 --epochs 50
   ```

5. Use cross-validation for more robust evaluation (optional):
   ```bash
   python train.py --task facial_features --model resnet18 --cross_val
   ```

### Evaluation

After training, evaluate each model using these commands:

1. Evaluate ResNet-18 for facial feature detection:
   ```bash
   python evaluate.py --task facial_features --model resnet18 --model_path output/facial_features_TIMESTAMP/best_model.pth
   ```

2. Evaluate ResNet-34 for facial feature detection:
   ```bash
   python evaluate.py --task facial_features --model resnet34 --model_path output/facial_features_TIMESTAMP/best_model.pth
   ```

3. Evaluate ResNet-18 for brain tissue loss detection:
   ```bash
   python evaluate.py --task brain_tissue_loss --model resnet18 --model_path output/brain_tissue_loss_TIMESTAMP/best_model.pth
   ```

4. Evaluate ResNet-34 for brain tissue loss detection:
   ```bash
   python evaluate.py --task brain_tissue_loss --model resnet34 --model_path output/brain_tissue_loss_TIMESTAMP/best_model.pth
   ```

5. Compare with reference model metrics:
   ```bash
   python evaluate.py --task facial_features --model resnet18 --model_path output/facial_features_TIMESTAMP/best_model.pth --reference_file reference_metrics.txt
   ```

Note: Replace TIMESTAMP with the actual timestamp created during the training process (format: YYYYMMDD_HHMMSS).

## Reference Values from Original Paper

For comparison, the reference model in the paper achieved:
- 95.49% accuracy in identifying recognizable facial features
- 97.63% accuracy in detecting the loss of brain tissue voxels

## Data Description

The dataset consists of skull-stripped brain MRI images processed with the Brain Extraction Tool (BET). The file naming pattern (asl_t1_XXX_bet_32.nii) indicates these are T1-weighted MRI scans that have undergone skull-stripping with various parameter settings, producing images with varying degrees of facial feature removal and potential brain tissue loss.