# 🏷️ Multi-Label Image Classification with Custom CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurimas13/multi-label-classification/blob/main/multilabel_classification.ipynb)

A custom neural network implementation for multi-label image classification built from scratch using PyTorch. This project demonstrates how to build, train, and evaluate a CNN for scenarios where images can belong to multiple classes simultaneously.

## 🌟 Features

- **Custom CNN Architecture**: Built from scratch with residual blocks and batch normalization
- **Multi-Label Support**: Binary Cross-Entropy loss for independent label predictions
- **Complete Training Pipeline**: Manual implementation without high-level `.fit()` methods
- **Advanced Optimization**: Mixed precision training, learning rate scheduling, and data augmentation
- **Comprehensive Metrics**: Hamming loss, exact match accuracy, precision, recall, and F1-score
- **GPU Acceleration**: Optimized for CUDA, MPS (Apple Silicon), and CPU
- **Visualization Tools**: Training curves, prediction analysis, and performance metrics

## 📊 Performance

| Metric | Random Baseline | Basic Model | Optimized Model |
|--------|----------------|-------------|-----------------|
| **Loss** | 0.693 | ~0.65 | **0.15-0.25** ✅ |
| **F1 Score** | 0.25 | ~0.33 | **0.75-0.85** ✅ |
| **Hamming Loss** | 0.50 | ~0.48 | **0.10-0.20** ✅ |
| **Exact Match** | 6.25% | ~13% | **40-60%** ✅ |

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - Free GPU!)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurimas13/multi-label-classification/blob/main/multilabel_classification.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → GPU
3. Run all cells sequentially

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/aurimas13/multi-label-classification.git
cd multi-label-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python multilabel_classification.py
```

### Option 3: Using Conda

```bash
# Create conda environment
conda create -n multilabel python=3.9
conda activate multilabel

# Install PyTorch (CPU)
conda install pytorch torchvision -c pytorch

# For CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Run training
python multilabel_classification.py
```

## 🏗️ Model Architecture

### Enhanced CNN with Residual Blocks
```
Input (224×224×3)
    ↓
Conv(3→64) + BatchNorm + ReLU + MaxPool
    ↓
ResBlock_1: 2×[Conv(64→64) + BatchNorm + ReLU] with skip connection
    ↓
ResBlock_2: 2×[Conv(64→128) + BatchNorm + ReLU] with skip connection
    ↓
ResBlock_3: 2×[Conv(128→256) + BatchNorm + ReLU] with skip connection
    ↓
ResBlock_4: 2×[Conv(256→512) + BatchNorm + ReLU] with skip connection
    ↓
Global Average Pooling
    ↓
FC(512→256) + ReLU + Dropout(0.3)
    ↓
FC(256→num_classes)
    ↓
Output (num_classes) [No activation - BCEWithLogitsLoss]
```

### Key Design Decisions

1. **Residual Connections**: Prevent vanishing gradients and enable deeper networks
2. **Global Average Pooling**: Reduces parameters and improves generalization
3. **BCEWithLogitsLoss**: Numerically stable for multi-label classification
4. **Batch Normalization**: Stabilizes training and accelerates convergence
5. **Dropout Regularization**: Prevents overfitting
6. **Mixed Precision Training**: 2-3× faster training on compatible GPUs

## 📈 Training Features

### Optimization Techniques
- **AdamW Optimizer**: Decoupled weight decay for better generalization
- **Cosine Annealing LR**: Smooth learning rate decay
- **Data Augmentation**: Random flips, rotations, color jitter, and affine transforms
- **Mixed Precision**: Automatic mixed precision for faster GPU training
- **Early Stopping**: Monitors validation F1 score

### Data Pipeline
- **Synthetic Patterns**: Learnable patterns (stripes, circles) for demonstration
- **Real Data Support**: Easy adaptation for Open Images or custom datasets
- **Efficient Loading**: Lazy loading with PyTorch DataLoader
- **Multi-worker Support**: Parallel data loading for faster training

## 📊 Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Loss** | Binary Cross-Entropy loss | < 0.3 |
| **Hamming Loss** | Fraction of wrong labels | < 0.2 |
| **Exact Match** | % samples with ALL labels correct | > 40% |
| **Precision** | Of predicted positives, how many correct? | > 0.7 |
| **Recall** | Of actual positives, how many found? | > 0.7 |
| **F1 Score** | Harmonic mean of Precision and Recall | > 0.7 |

## 🗂️ Project Structure

```
multi-label-classification/
│
├── multilabel_classification.py   # Basic implementation
├── multilabel_optimized.py        # Enhanced version with optimizations
├── multilabel_classification.ipynb # Jupyter notebook for Colab
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT license
├── .gitignore                      # Git ignore rules
│
├── models/                         # Saved models directory
│   ├── best_model.pth             # Best performing model
│   └── checkpoints/               # Training checkpoints
│
├── results/                        # Training results
│   ├── training_history.png       # Loss and metric curves
│   └── prediction_analysis.png    # Sample predictions
│
└── data/                          # Dataset directory (create for real data)
    ├── images/                    # Image files
    └── annotations/               # Label files
```

## 🎯 Usage Examples

### Basic Training
```python
from multilabel_classification import train_model

# Train with default configuration
model, history, metrics = train_model()
```

### Custom Configuration
```python
config = {
    'num_classes': 10,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'dropout': 0.4
}

model, history, metrics = train_model(config)
```

### Inference on New Images
```python
import torch
from PIL import Image

# Load trained model
model = CustomCNN(num_classes=4)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predict on new image
image = Image.open('test_image.jpg')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.sigmoid(output)
    predictions = (probabilities > 0.5).int()
```

## 🔧 Customization

### Using Real Data (Open Images Example)

```python
# 1. Download Open Images subset
!pip install openimages
!oi_download_dataset --dataset train --classes Cat,Dog,Car,Tree --limit 1000

# 2. Modify dataset class
class RealImageDataset(Dataset):
    def __init__(self, image_dir, annotations_file, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        labels = self.annotations.iloc[idx][1:].values.astype(float)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(labels, dtype=torch.float32)
```

### Hyperparameter Tuning

```python
# Grid search example
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64, 128]
dropout_rates = [0.2, 0.3, 0.5]

best_f1 = 0
best_config = {}

for lr in learning_rates:
    for bs in batch_sizes:
        for dropout in dropout_rates:
            config = {
                'learning_rate': lr,
                'batch_size': bs,
                'dropout': dropout
            }
            model, history, metrics = train_model(config)
            
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_config = config
```

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 2GB free space
- **GPU**: Optional (but recommended)

### Recommended Setup
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **RAM**: 16GB+
- **CUDA**: 11.8 or higher
- **Storage**: 10GB for datasets

### Training Time Estimates
| Hardware | Time per Epoch | 50 Epochs |
|----------|---------------|-----------|
| CPU (i5/i7) | ~60s | ~50 min |
| GPU (GTX 1060) | ~10s | ~8 min |
| GPU (RTX 3080) | ~5s | ~4 min |
| Colab (T4) | ~8s | ~7 min |
| Apple M1/M2 | ~15s | ~12 min |

## 📝 Results Visualization

The training pipeline generates comprehensive visualizations:

1. **Training History**: Loss curves, F1 score progression, learning rate schedule
2. **Metrics Evolution**: Precision/Recall trade-off, overfitting monitor
3. **Prediction Analysis**: Sample predictions with confidence scores
4. **Confusion Matrices**: Per-class performance analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Multi-Label Classification: An Overview](https://www.sciencedirect.com/science/article/pii/S0031320313001471)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Aurimas Aleksandras**
- GitHub: [@aurimas13](https://github.com/aurimas13)
- LinkedIn: [Aurimas Aleksandras](https://www.linkedin.com/in/aurimas-aleksandras/)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Google Colab for free GPU resources
- Open Images Dataset creators
- Multi-label classification research community

## 📊 Citation

If you use this code in your research, please cite:

```bibtex
@software{aurimas2024multilabel,
  author = {Aleksandras, Aurimas},
  title = {Multi-Label Image Classification with Custom CNN},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aurimas13/multi-label-classification}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=aurimas13/multi-label-classification&type=Date)](https://star-history.com/#aurimas13/multi-label-classification&Date)

---

<p align="center">
Made with ❤️ for the deep learning community
</p>