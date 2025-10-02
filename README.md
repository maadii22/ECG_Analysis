# ECG Classification Project

A complete ECG classification pipeline using deep learning to classify ECG images as Normal or Abnormal.

## 🎯 Project Overview

This project implements an end-to-end ECG classification system that:
- Creates high-quality synthetic ECG image datasets
- Trains a CNN model for binary classification
- Achieves high accuracy with proper data preprocessing
- Provides complete training and evaluation pipeline

## 📁 Project Structure

```
ECG analysis/
├── run_pipeline.py           # Main pipeline runner
├── create_dataset.py         # Create synthetic ECG images
├── train_model.py           # Train with synthetic data
├── download_real_datasets.py # Download real ECG datasets
├── train_with_real_data.py  # Train with real data
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── USAGE.md                 # Usage instructions
├── datasets/                # Data directory
│   ├── ecg_images/         # ECG images
│   │   ├── NORMAL/        # Normal ECG images
│   │   └── ABNORMAL/      # Abnormal ECG images
│   ├── ecg_metadata.csv  # Image labels and paths
│   └── models/            # Trained models
└── src/                   # Source code
    └── utils.py           # Utility functions
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python3 run_pipeline.py
```
This will give you two options:
- **Option 1**: Use synthetic ECG images (fast, for testing)
- **Option 2**: Download and use real ECG datasets (slower, better accuracy)

### 3. Alternative: Step by Step
```bash
# For synthetic data (fast)
python3 create_dataset.py
python3 train_model.py

# For real data (better accuracy)
python3 download_real_datasets.py
python3 train_with_real_data.py
```

## 📊 Expected Results

- **Dataset**: 400 ECG images (200 normal + 200 abnormal)
- **Training Time**: 10-30 minutes
- **Expected Accuracy**: 85-95%
- **Model**: CNN with batch normalization and dropout

## 🔧 Key Features

- ✅ **High-quality synthetic ECG patterns** with realistic waveforms
- ✅ **Proper CNN architecture** with batch normalization and dropout
- ✅ **Data augmentation** and regularization
- ✅ **Early stopping** and learning rate reduction
- ✅ **Real image processing** instead of dummy data
- ✅ **Complete training pipeline** with evaluation

## 📈 Model Architecture

The CNN model includes:
- **3 Convolutional blocks** with batch normalization
- **Global average pooling** for better generalization
- **Dense layers** with dropout for regularization
- **Binary classification** output (Normal/Abnormal)

## 🎯 Usage

### Create Dataset
```python
from create_dataset import create_ecg_dataset
df = create_ecg_dataset()
```

### Train Model
```python
from train_model import train_model
accuracy, history = train_model()
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 🔍 Troubleshooting

### Low Accuracy Issues
- Ensure you're using real ECG images (not dummy data)
- Check that images are properly loaded and preprocessed
- Verify the model architecture is appropriate

### Memory Issues
- Reduce batch size in training
- Use smaller image sizes
- Reduce number of training samples

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!# ECG_Analysis
# ECG_Analysis
# ECG_Analysis
# ECG_Analysis
# ECG_Analysis
