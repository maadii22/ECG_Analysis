#!/usr/bin/env python3
"""
Utility functions for ECG classification project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import cv2
from PIL import Image


def load_image_rgb(path, target_size=None):
    """Load an image from disk, convert to RGB, optionally resize to target_size (width, height).

    Returns a numpy array with dtype uint8 or float32 depending on use.
    """
    with Image.open(path) as im:
        im = im.convert('RGB')
        if target_size is not None:
            im = im.resize(target_size, resample=Image.BILINEAR)
        return np.asarray(im)


def focal_loss(alpha=0.25, gamma=2.0):
    """Focal loss implementation"""
    def fl(y_true, y_pred):
        # Convert to one-hot encoding with correct number of classes (2 for binary)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return fl


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot AUC
    axes[1].plot(history.history['auc'], label='Training AUC')
    axes[1].plot(history.history['val_auc'], label='Validation AUC')
    axes[1].set_title('Model AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_class_distribution(df, column='label', save_path=None):
    """Plot class distribution"""
    plt.figure(figsize=(12, 6))
    class_counts = df[column].value_counts()
    
    bars = plt.bar(range(len(class_counts)), class_counts.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    plt.close()


def plot_sample_ecg_images(df, img_dir, n_samples=6, save_path=None):
    """Plot sample ECG images"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(n_samples, len(df))):
        img_path = df.iloc[i]['path']
        label = df.iloc[i]['label']
        
        if os.path.exists(img_path):
            img = load_image_rgb(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample ECG images saved to: {save_path}")
    
    plt.close()


def create_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Create Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model([model.inputs], 
                                     [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def plot_gradcam_visualization(model, test_df, class_names, img_size=(224, 224), 
                              n_samples=6, save_path=None):
    """Create Grad-CAM visualization"""
    # Find last conv layer
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    
    if last_conv_name is None:
        print("No convolutional layer found for Grad-CAM")
        return
    
    print(f"Using layer for Grad-CAM: {last_conv_name}")
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(n_samples, len(test_df))):
        img_path = test_df.iloc[i]['path']
        true_label = test_df.iloc[i]['label']
        
        if not os.path.exists(img_path):
            continue
        
        # Load and preprocess image
        img = tf.keras.utils.load_img(img_path, target_size=img_size)
        x = tf.keras.utils.img_to_array(img) / 255.0
        x_input = np.expand_dims(x, axis=0)
        
        # Get prediction
        preds = model.predict(x_input, verbose=0)
        pred_class = np.argmax(preds[0])
        pred_label = class_names[pred_class]
        confidence = preds[0][pred_class]
        
        # Create Grad-CAM heatmap
        heatmap = create_gradcam_heatmap(x_input, model, last_conv_name)
        heatmap = np.uint8(255 * heatmap)
        
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (img_size[1], img_size[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        superimposed_img = heatmap * 0.4 + np.uint8(x * 255)
        
        # Plot original image
        axes[0, i].imshow(x)
        axes[0, i].set_title(f'Original\nTrue: {true_label}')
        axes[0, i].axis('off')
        
        # Plot Grad-CAM
        axes[1, i].imshow(superimposed_img.astype('uint8'))
        axes[1, i].set_title(f'Grad-CAM\nPred: {pred_label}\nConf: {confidence:.3f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to: {save_path}")
    
    plt.close()


def save_results_summary(results, save_path):
    """Save results summary to text file"""
    with open(save_path, 'w') as f:
        f.write("ECG Classification Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance:\n")
        f.write(f"Macro F1 Score: {results.get('macro_f1', 'N/A')}\n")
        f.write(f"Macro AUC Score: {results.get('macro_auc', 'N/A')}\n")
        f.write(f"Accuracy: {results.get('accuracy', 'N/A')}\n\n")
        
        f.write("Classification Report:\n")
        f.write(str(results.get('classification_report', 'N/A')) + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(results.get('confusion_matrix', 'N/A')) + "\n")
    
    print(f"Results summary saved to: {save_path}")


def create_model_summary(model, save_path):
    """Create and save model summary"""
    with open(save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model summary saved to: {save_path}")


def calculate_class_weights(df, label_column='label_id'):
    """Calculate class weights for imbalanced dataset"""
    class_counts = df[label_column].value_counts().to_dict()
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    
    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[int(class_id)] = total / (n_classes * count)
    
    return class_weights


def preprocess_ecg_signal(signal, target_length=5000):
    """Preprocess ECG signal"""
    # Standardize length
    L = signal.shape[0]
    if L < target_length:
        pad = target_length - L
        left = pad // 2
        right = pad - left
        signal = np.pad(signal, ((left, right), (0, 0)), mode='constant')
    else:
        start = (L - target_length) // 2
        signal = signal[start:start + target_length, :]
    
    # Normalize per-lead
    signal = (signal - np.mean(signal, axis=0, keepdims=True)) / (np.std(signal, axis=0, keepdims=True) + 1e-8)
    
    return signal


def create_data_augmentation_pipeline():
    """Create data augmentation pipeline"""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])
    
    return data_augmentation


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """Plot ROC curves for each class"""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    
    plt.close()


def create_feature_importance_plot(model, layer_name, save_path=None):
    """Create feature importance plot from model layer"""
    try:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()[0]  # Get weights from the layer
        
        if len(weights.shape) == 2:  # Dense layer
            importance = np.abs(weights).mean(axis=1)
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(importance)), importance)
            plt.title(f'Feature Importance - {layer_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Average Absolute Weight')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature importance plot saved to: {save_path}")
            
            plt.close()
        else:
            print(f"Layer {layer_name} weights shape not suitable for feature importance plot")
    
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")


def save_predictions_csv(y_true, y_pred, y_pred_proba, class_names, save_path):
    """Save predictions to CSV file"""
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_class_name': [class_names[i] for i in y_true],
        'predicted_class_name': [class_names[i] for i in y_pred]
    })
    
    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = y_pred_proba[:, i]
    
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")


def create_ensemble_prediction(models, X_test):
    """Create ensemble prediction from multiple models"""
    predictions = []
    
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred


def plot_learning_curves(history, save_path=None):
    """Plot detailed learning curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC
    axes[0, 1].plot(history.history['auc'], label='Training AUC')
    axes[0, 1].plot(history.history['val_auc'], label='Validation AUC')
    axes[0, 1].set_title('Model AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Additional metrics if available
    other_metrics = [k for k in history.history.keys() if k not in ['loss', 'val_loss', 'auc', 'val_auc', 'lr']]
    if other_metrics:
        metric = other_metrics[0]
        axes[1, 1].plot(history.history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            axes[1, 1].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        axes[1, 1].set_title(f'Model {metric}')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel(metric)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    plt.close()
