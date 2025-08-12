"""
Multi-Label Image Classification Using Custom Neural Network
=============================================================
This implementation creates a custom CNN from scratch for multi-label classification.
Each image can have multiple labels (e.g., an image can contain both 'cat' and 'indoor').

Key Design Decisions:
1. Custom CNN architecture (not transfer learning)
2. Binary Cross-Entropy loss for multi-label classification
3. Manual training loop (no .fit() methods)
4. Comprehensive validation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, accuracy_score
import os
import json
from typing import Tuple, List, Dict
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
# This ensures consistent results across runs
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration - use GPU if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MultiLabelDataset(Dataset):
    """
    Custom Dataset for Multi-Label Classification
    
    Why custom dataset?
    - Handles multi-label targets (multiple classes per image)
    - Applies data augmentation on-the-fly
    - Memory efficient - loads images as needed
    """
    
    def __init__(self, image_paths: List[str], labels: List[List[int]], 
                 transform=None, num_classes: int = 4):
        """
        Args:
            image_paths: List of paths to images
            labels: List of multi-hot encoded labels for each image
            transform: torchvision transforms for preprocessing
            num_classes: Total number of possible classes
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.num_classes = num_classes
        
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a sample
        
        Why this approach?
        - Lazy loading saves memory
        - Transforms applied on-the-fly for augmentation
        - Returns multi-hot encoded labels as float tensor for BCE loss
        """
        # For demo purposes, create synthetic image data
        # In real scenario, you would load: img = Image.open(self.image_paths[idx])
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            img = self.transform(img)
        
        # Convert multi-hot label to tensor
        # Float type required for BCEWithLogitsLoss
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, label

class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network for Multi-Label Classification
    
    Architecture Design Rationale:
    - 4 convolutional blocks for hierarchical feature extraction
    - Progressive channel increase (3→32→64→128→256) for learning complex patterns
    - Batch normalization for stable training and faster convergence
    - Dropout for regularization to prevent overfitting
    - Global Average Pooling instead of flatten to reduce parameters
    - No sigmoid in final layer (using BCEWithLogitsLoss for numerical stability)
    """
    
    def __init__(self, num_classes: int = 4):
        super(CustomCNN, self).__init__()
        
        # BLOCK 1: Initial feature extraction
        # Conv 3→32: Learn low-level features (edges, colors)
        # Kernel size 3x3: Standard choice, captures local patterns
        # Padding=1: Maintains spatial dimensions with stride=1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalizes activations, prevents internal covariate shift
        self.relu1 = nn.ReLU(inplace=True)  # Non-linearity, inplace saves memory
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces spatial dims by 2x, provides translation invariance
        
        # BLOCK 2: Mid-level feature extraction
        # Conv 32→64: Increases capacity to learn more complex patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # BLOCK 3: High-level feature extraction
        # Conv 64→128: Further increases model capacity
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # BLOCK 4: Abstract feature extraction
        # Conv 128→256: Final convolutional features
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Global Average Pooling: Reduces each channel to single value
        # Advantages over Flatten+FC:
        # - Fewer parameters (prevents overfitting)
        # - Enforces correspondence between feature maps and classes
        # - More robust to spatial translations
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout: Randomly zeros 50% of neurons during training
        # Prevents co-adaptation of neurons, improves generalization
        self.dropout = nn.Dropout(0.5)
        
        # Classification head
        # 256→512: Intermediate layer for non-linear decision boundary
        self.fc1 = nn.Linear(256, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # Final layer: 512→num_classes
        # No activation here - BCEWithLogitsLoss applies sigmoid internally
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Input shape: (batch_size, 3, 224, 224)
        Output shape: (batch_size, num_classes)
        """
        # Block 1: (B,3,224,224) → (B,32,112,112)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        
        # Block 2: (B,32,112,112) → (B,64,56,56)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # Block 3: (B,64,56,56) → (B,128,28,28)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Block 4: (B,128,28,28) → (B,256,28,28)
        x = self.relu4(self.bn4(self.conv4(x)))
        
        # Global pooling: (B,256,28,28) → (B,256,1,1)
        x = self.global_pool(x)
        
        # Flatten: (B,256,1,1) → (B,256)
        x = x.view(x.size(0), -1)
        
        # Classifier with dropout
        x = self.dropout(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # Raw logits for BCEWithLogitsLoss
        
        return x

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-label classification
    
    Why these metrics?
    - Hamming Loss: Fraction of wrong labels (lower is better)
    - Exact Match: Strict accuracy (all labels must match)
    - Precision/Recall/F1: Per-label performance averaged
    - These give complete picture of multi-label performance
    """
    # Convert logits to predictions using sigmoid + threshold
    # Sigmoid converts logits to probabilities [0,1]
    # Threshold 0.5 is standard for binary decisions
    preds = (torch.sigmoid(predictions) > threshold).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Hamming loss: Average fraction of incorrect labels
    h_loss = hamming_loss(targets, preds)
    
    # Exact match ratio: Percentage of samples with ALL labels correct
    exact_match = accuracy_score(targets, preds)
    
    # Per-label metrics averaged across all labels
    # 'samples' averaging: Calculate metrics for each sample, then average
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average='samples', zero_division=0
    )
    
    return {
        'hamming_loss': h_loss,
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """
    Train the model for one epoch
    
    Training Process:
    1. Forward pass: Compute predictions
    2. Calculate loss using BCEWithLogitsLoss
    3. Backward pass: Compute gradients
    4. Update weights using optimizer
    """
    model.train()  # Enable dropout and batch norm training behavior
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients from previous step
        # PyTorch accumulates gradients by default
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(images)
        
        # Calculate loss
        # BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights based on gradients
        # Optimizer step applies the chosen optimization algorithm
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    # Return average loss over all batches
    return total_loss / len(dataloader)

def validate_epoch(model: nn.Module, dataloader: DataLoader, 
                   criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model on validation set
    
    Why validation?
    - Monitor overfitting (training loss ↓ but validation loss ↑)
    - Track real performance on unseen data
    - Make decisions about early stopping
    """
    model.eval()  # Disable dropout and use batch norm running stats
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # Disable gradient computation for efficiency
    # We don't need gradients during validation
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions for metric calculation
            all_predictions.append(outputs)
            all_targets.append(labels)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics

def create_synthetic_data(num_samples: int = 1000, num_classes: int = 4) -> Tuple:
    """
    Create synthetic multi-label data for demonstration
    
    In real scenario, you would:
    1. Download dataset (e.g., subset of Open Images)
    2. Parse annotations for multi-label targets
    3. Create train/val/test splits
    """
    image_paths = [f"image_{i}.jpg" for i in range(num_samples)]
    
    # Create multi-hot encoded labels
    # Each sample can have 1-3 labels active
    labels = []
    for _ in range(num_samples):
        # Random number of active labels per image
        num_active = np.random.randint(1, min(4, num_classes + 1))
        active_classes = np.random.choice(num_classes, num_active, replace=False)
        
        # Multi-hot encoding: [0,1,0,1] means classes 1 and 3 are present
        label = np.zeros(num_classes)
        label[active_classes] = 1
        labels.append(label.tolist())
    
    # 70-15-15 train-val-test split
    train_split = int(0.7 * num_samples)
    val_split = int(0.85 * num_samples)
    
    return (image_paths[:train_split], labels[:train_split],
            image_paths[train_split:val_split], labels[train_split:val_split],
            image_paths[val_split:], labels[val_split:])

def inference_on_sample(model: nn.Module, dataloader: DataLoader, 
                        class_names: List[str], threshold: float = 0.5):
    """
    Perform inference on a random sample and visualize results
    
    Shows:
    - Input image
    - Predicted labels with confidence scores
    - Ground truth labels
    """
    model.eval()
    
    # Get a random batch
    images, labels = next(iter(dataloader))
    
    # Select first image from batch
    image = images[0:1].to(device)
    true_label = labels[0]
    
    with torch.no_grad():
        output = model(image)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    print("\n=== Inference on Sample Image ===")
    print(f"Predicted probabilities: {probs}")
    
    # Get predicted labels based on threshold
    predicted_labels = [class_names[i] for i, p in enumerate(probs) if p > threshold]
    true_labels = [class_names[i] for i, l in enumerate(true_label) if l == 1]
    
    print(f"Predicted labels: {predicted_labels}")
    print(f"True labels: {true_labels}")
    
    # Calculate confidence for each predicted label
    for i, class_name in enumerate(class_names):
        if probs[i] > threshold:
            print(f"  {class_name}: {probs[i]:.2%} confidence")

def main():
    """
    Main training pipeline
    
    Complete workflow:
    1. Data preparation with augmentation
    2. Model initialization
    3. Training loop with validation
    4. Final evaluation
    5. Inference demonstration
    """
    
    # Hyperparameters - these control training behavior
    NUM_CLASSES = 4  # Number of possible labels
    BATCH_SIZE = 32  # Samples per gradient update (32 is good balance of speed/stability)
    LEARNING_RATE = 0.001  # Step size for weight updates (0.001 is standard for Adam)
    NUM_EPOCHS = 10  # Complete passes through dataset
    
    # Class names for visualization
    class_names = ['Cat', 'Dog', 'Indoor', 'Outdoor']
    
    print("=== Multi-Label Image Classification ===")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}\n")
    
    # Data preparation
    print("Preparing synthetic dataset...")
    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = create_synthetic_data()
    
    # Data augmentation for training set only
    # Why augmentation?
    # - Increases effective dataset size
    # - Improves generalization
    # - Reduces overfitting
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        transforms.RandomRotation(10),  # Random rotation ±10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variations
        transforms.ToTensor(),  # Convert PIL to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                           std=[0.229, 0.224, 0.225])   # Normalize to [-1, 1] range
    ])
    
    # Validation/Test transform - no augmentation, only preprocessing
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultiLabelDataset(train_imgs, train_labels, train_transform, NUM_CLASSES)
    val_dataset = MultiLabelDataset(val_imgs, val_labels, val_transform, NUM_CLASSES)
    test_dataset = MultiLabelDataset(test_imgs, test_labels, val_transform, NUM_CLASSES)
    
    # Create dataloaders for batched training
    # shuffle=True for training: randomizes order each epoch (prevents memorization)
    # shuffle=False for val/test: consistent order for reproducibility
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function: BCEWithLogitsLoss for multi-label classification
    # Why BCEWithLogitsLoss?
    # - Numerically stable (combines sigmoid + BCE)
    # - Suitable for multi-label (independent binary classifications)
    # - Each output neuron represents probability of one label
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer: Adam
    # Why Adam?
    # - Adaptive learning rates per parameter
    # - Combines benefits of RMSprop and momentum
    # - Works well out-of-the-box for most problems
    # - Less sensitive to hyperparameter tuning than SGD
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history for visualization
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    print("\n=== Starting Training ===")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_f1_scores.append(val_metrics['f1_score'])
        
        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics:")
        print(f"    - Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"    - Exact Match: {val_metrics['exact_match']:.4f}")
        print(f"    - Precision: {val_metrics['precision']:.4f}")
        print(f"    - Recall: {val_metrics['recall']:.4f}")
        print(f"    - F1 Score: {val_metrics['f1_score']:.4f}")
    
    # Final test evaluation
    print("\n=== Final Test Evaluation ===")
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics:")
    print(f"  - Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print(f"  - Exact Match: {test_metrics['exact_match']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall: {test_metrics['recall']:.4f}")
    print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
    
    # Inference on sample
    inference_on_sample(model, test_loader, class_names)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\n✓ Training history saved to 'training_history.png'")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': NUM_EPOCHS,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_metrics': val_metrics
    }, 'multilabel_model.pth')
    print("✓ Model saved to 'multilabel_model.pth'")

if __name__ == "__main__":
    main()
