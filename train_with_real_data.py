#!/usr/bin/env python3
"""
Train ECG Classification Model with Real Datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import glob
from project_config import DATA_ROOT
from PIL import Image

def find_ecg_images(data_dir):
    """Find all ECG images in the downloaded datasets"""
    print("🔍 Searching for ECG images in downloaded datasets...")
    
    image_paths = []
    labels = []
    
    # Search patterns for different dataset structures
    search_patterns = [
        f"{data_dir}/ECG_Cardiac_Images/**/*.png",
        f"{data_dir}/ECG_Cardiac_Images/**/*.jpg",
        f"{data_dir}/ECG_Cardiac_Images/**/*.jpeg",
        f"{data_dir}/ECG_Arrhythmia_Images/**/*.png",
        f"{data_dir}/ECG_Arrhythmia_Images/**/*.jpg",
        f"{data_dir}/ECG_Arrhythmia_Images/**/*.jpeg",
        f"{data_dir}/ECG_Heartbeat/**/*.png",
        f"{data_dir}/ECG_Heartbeat/**/*.jpg",
        f"{data_dir}/ECG_Heartbeat/**/*.jpeg"
    ]
    
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            image_paths.append(file_path)
            
            # Determine label based on path and filename
            path_lower = file_path.lower()
            if any(keyword in path_lower for keyword in ['normal', 'n', '0', 'healthy']):
                labels.append(0)  # Normal
            elif any(keyword in path_lower for keyword in ['abnormal', 'a', '1', 'disease', 'arrhythmia']):
                labels.append(1)  # Abnormal
            else:
                # Default to abnormal if unclear
                labels.append(1)
    
    print(f"Found {len(image_paths)} ECG images")
    return image_paths, labels

def load_and_preprocess_images(image_paths, labels, img_size=(224, 224), max_samples=1000):
    """Load and preprocess ECG images"""
    print(f"Loading and preprocessing {min(len(image_paths), max_samples)} images...")
    
    images = []
    processed_labels = []
    
    # Limit samples for faster processing
    sample_size = min(len(image_paths), max_samples)
    indices = np.random.choice(len(image_paths), sample_size, replace=False)
    
    for i, idx in enumerate(indices):
        try:
            img_path = image_paths[idx]
            label = labels[idx]
            
            # Load + resize with PIL for robustness
            with Image.open(img_path) as im:
                im = im.convert('RGB')
                im = im.resize(img_size, resample=Image.BILINEAR)
                img_arr = np.asarray(im, dtype=np.float32) / 255.0
            img_normalized = img_arr
            
            images.append(img_normalized)
            processed_labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{sample_size} images")
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return np.array(images), np.array(processed_labels)

def build_improved_model(input_shape):
    """Build an improved CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Global average pooling
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile with better optimizer and learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = DATA_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'training_history_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_with_real_data():
    """Train model with real ECG datasets"""
    print("🏥 Training with Real ECG Datasets")
    print("=" * 50)
    
    # Check if datasets exist
    data_dir = DATA_ROOT
    
    if not data_dir.exists():
        print("❌ Datasets directory not found!")
        print("Please run download_real_datasets.py first")
        return None
    
    # Find ECG images
    image_paths, labels = find_ecg_images(str(data_dir))
    
    if len(image_paths) == 0:
        print("❌ No ECG images found!")
        print("Please run download_real_datasets.py first")
        return None
    
    print(f"✅ Found {len(image_paths)} ECG images")
    
    # Load and preprocess images
    images, processed_labels = load_and_preprocess_images(image_paths, labels, max_samples=2000)
    
    if len(images) == 0:
        print("❌ No images loaded successfully!")
        return None
    
    print(f"✅ Successfully loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Normal samples: {np.sum(processed_labels == 0)}")
    print(f"Abnormal samples: {np.sum(processed_labels == 1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, processed_labels, test_size=0.2, random_state=42, stratify=processed_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📊 Data Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Build model
    input_shape = images[0].shape
    model = build_improved_model(input_shape)
    
    print(f"🏗️ Model built with input shape: {input_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            str(data_dir / "models" / "best_model_real_data.h5"),
            save_best_only=True, monitor='val_accuracy'
        )
    ]
    
    # Train model
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Final Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    models_dir = data_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model.save(str(models_dir / "final_model_real_data.h5"))
    
    # Plot training history
    plot_training_history(history)
    
    return test_acc, history

if __name__ == "__main__":
    train_with_real_data()
