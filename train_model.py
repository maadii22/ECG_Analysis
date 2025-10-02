#!/usr/bin/env python3
"""
Train ECG Classification Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from project_config import DATA_ROOT
from PIL import Image

def load_and_preprocess_images(df, img_size=(224, 224)):
    """Load and preprocess ECG images using PIL for robust resizing/conversion"""
    print("Loading and preprocessing images...")
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            img_path = row['path']
            # Use PIL to load and resize consistently
            with Image.open(img_path) as im:
                im = im.convert('RGB')
                im = im.resize(img_size, resample=Image.BILINEAR)
                img_arr = np.asarray(im, dtype=np.float32) / 255.0

            images.append(img_arr)
            labels.append(int(row['label_id']))

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} images")

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    return np.array(images), np.array(labels)

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
    plt.savefig(results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_model():
    """Train model with real ECG images"""
    print("Training model with real ECG images...")
    
    # Load dataset
    data_dir = DATA_ROOT
    metadata_path = data_dir / "ecg_metadata.csv"
    df = pd.read_csv(metadata_path)
    
    print(f"Loading {len(df)} images...")
    
    # Load and preprocess images
    images, labels = load_and_preprocess_images(df)
    
    if len(images) == 0:
        print("No images loaded successfully!")
        return None
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    input_shape = images[0].shape
    model = build_improved_model(input_shape)
    
    print(f"Model built with input shape: {input_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            str(data_dir / "models" / "best_model.h5"),
            save_best_only=True, monitor='val_accuracy'
        )
    ]
    
    # Train model
    print("Training model...")
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
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    models_dir = data_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model.save(str(models_dir / "final_model.h5"))
    
    # Plot training history
    plot_training_history(history)
    
    return test_acc, history

if __name__ == "__main__":
    train_model()
