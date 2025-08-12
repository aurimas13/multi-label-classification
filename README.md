# Multi-Label Image Classification with Custom CNN

A custom neural network implementation for multi-label image classification built from scratch using PyTorch.

## 🎯 Project Overview

This project implements a custom Convolutional Neural Network (CNN) for multi-label classification where each image can belong to multiple classes simultaneously. The implementation includes:

- **Custom CNN Architecture**: 4-layer convolutional network with batch normalization and dropout
- **Complete Training Loop**: Manual implementation without high-level `.fit()` methods
- **Multi-Label Support**: Uses Binary Cross-Entropy loss for independent label predictions
- **Comprehensive Metrics**: Hamming loss, exact match accuracy, precision, recall, and F1-score
- **Inference Pipeline**: Test evaluation and single-sample prediction demonstration

## 🏗️ Architecture Details

### Model Architecture
```
Input (224x224x3)
    ↓
Conv Block 1: Conv(3→32) + BatchNorm + ReLU + MaxPool
    ↓
Conv Block 2: Conv(32→64) + BatchNorm + ReLU + MaxPool
    ↓
Conv Block 3: Conv(64→128) + BatchNorm + ReLU + MaxPool
    ↓
Conv Block 4: Conv(128→256) + BatchNorm + ReLU
    ↓
Global Average Pooling
    ↓
FC Layer 1: Linear(256→512) + ReLU + Dropout(0.5)
    ↓
FC Layer 2: Linear(512→num_classes)
    ↓
Output (num_classes)
```

### Key Design Decisions

1. **BCEWithLogitsLoss**: Combines sigmoid activation with BCE loss for numerical stability
2. **Global Average Pooling**: Reduces parameters and improves generalization vs traditional flatten
3. **Batch Normalization**: Stabilizes training and accelerates convergence
4. **Dropout (0.5)**: Prevents overfitting by randomly deactivating neurons during training
5. **Adam Optimizer**: Adaptive learning rates with momentum for efficient optimization

## 📊 Metrics Explained

- **Hamming Loss**: Fraction of wrong labels (lower is better)
- **Exact Match**: Percentage of samples with ALL labels correctly predicted
- **Precision**: Of predicted positive labels, how many were correct?
- **Recall**: Of actual positive labels, how many were found?
- **F1-Score**: Harmonic mean of precision and recall

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

### Running the Code
```bash
python multilabel_classification.py
```

### Expected Output
- Training progress with loss per epoch
- Validation metrics after each epoch
- Final test set evaluation
- Sample inference with predicted labels
- Training history plots saved as `training_history.png`
- Trained model saved as `multilabel_model.pth`

## 📁 Project Structure
```
multi-label-classification/
│
├── multilabel_classification.py  # Main implementation
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
│
├── training_history.png           # Generated after training
└── multilabel_model.pth           # Saved model checkpoint
```

## 💡 Usage with Real Data

To use with real image data (e.g., Open Images dataset):

1. **Download Dataset**:
```python
# Example for Open Images subset
# Download images and annotations
# Parse multi-label annotations
```

2. **Modify Dataset Class**:
```python
def __getitem__(self, idx):
    # Replace synthetic data with:
    img = Image.open(self.image_paths[idx]).convert('RGB')
    # Rest remains the same
```

3. **Update Data Loading**:
```python
# Parse real annotations
# Create train/val/test splits
# Pass real paths and labels to MultiLabelDataset
```

## 🔧 Customization Options

### Hyperparameters
- `NUM_CLASSES`: Number of possible labels (default: 4)
- `BATCH_SIZE`: Samples per gradient update (default: 32)
- `LEARNING_RATE`: Optimizer step size (default: 0.001)
- `NUM_EPOCHS`: Training iterations (default: 10)

### Model Architecture
- Adjust convolutional layers depths
- Modify dropout rates
- Change pooling strategies
- Add/remove layers

### Data Augmentation
- Adjust augmentation probabilities
- Add new transformations (crop, blur, etc.)
- Modify normalization statistics

## 📈 Performance Considerations

1. **GPU Usage**: Automatically uses CUDA if available
2. **Memory Efficiency**: Lazy loading of images
3. **Batch Processing**: Efficient mini-batch gradient descent
4. **Mixed Precision**: Can add `torch.cuda.amp` for faster training

## 🧪 Testing

The code includes comprehensive testing:
- Validation after each epoch
- Final test set evaluation
- Single sample inference demonstration
- Metric calculation and visualization

## 📝 Notes

- The current implementation uses **synthetic data** for demonstration
- For production use, replace with actual image dataset
- Model performance depends on data quality and quantity
- Consider transfer learning for better performance on small datasets

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use in your projects

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Multi-label classification research community
- Open Images dataset creators (when using real data)