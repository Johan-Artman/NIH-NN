"""
Demo script to show the NIH X-ray classification system without requiring the full dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.nih_cnn import MedicalImageCNN, TransferLearningCNN, compile_model


def create_synthetic_data(num_samples=1000):
    """
    Create synthetic X-ray-like data for demonstration purposes.
    
    Args:
        num_samples (int): Number of synthetic samples to create
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    print(f"[INFO] Creating {num_samples} synthetic X-ray samples...")
    
    # Create synthetic X-ray-like images (grayscale patterns converted to RGB)
    X = np.random.rand(num_samples, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    
    # Add some structure to make them look more like X-rays
    for i in range(num_samples):
        # Add circular/oval shapes (ribcage-like)
        center_y, center_x = config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2
        y, x = np.ogrid[:config.IMAGE_SIZE, :config.IMAGE_SIZE]
        
        # Create oval shape
        mask = ((x - center_x) / (config.IMAGE_SIZE * 0.3))**2 + ((y - center_y) / (config.IMAGE_SIZE * 0.4))**2 < 1
        X[i][mask] *= 0.3  # Darker inside (lung area)
        
        # Add some noise
        X[i] += np.random.normal(0, 0.1, X[i].shape)
        X[i] = np.clip(X[i], 0, 1)
    
    # Create synthetic labels
    num_classes = len(config.CLASSES)
    y = np.random.randint(0, num_classes, num_samples)
    
    # Convert to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y, num_classes)
    
    # Split into train and validation
    split_idx = int(0.8 * num_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_onehot[:split_idx], y_onehot[split_idx:]
    
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def demonstrate_models():
    """
    Demonstrate different model architectures.
    """
    print("\n[INFO] Demonstrating model architectures...")
    
    num_classes = len(config.CLASSES)
    
    # Only demonstrate custom CNN to avoid downloading pre-trained weights
    model = MedicalImageCNN.build(
        width=config.IMAGE_SIZE,
        height=config.IMAGE_SIZE,
        depth=3,
        classes=num_classes
    )
    
    print(f"\n[INFO] Custom CNN Architecture:")
    print(f"  - Total parameters: {model.count_params():,}")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Output shape: {model.output_shape}")
    print(f"  - Model layers: {len(model.layers)}")
    
    print("\n[INFO] Transfer learning models (VGG16, ResNet50) are also available")
    print("      but require internet access to download pre-trained weights.")


def run_mini_training():
    """
    Run a mini training session on synthetic data.
    """
    print("\n[INFO] Running mini training demonstration...")
    
    # Create synthetic data
    X_train, y_train, X_val, y_val = create_synthetic_data(500)
    
    # Build and compile model
    model = MedicalImageCNN.build(
        width=config.IMAGE_SIZE,
        height=config.IMAGE_SIZE,
        depth=3,
        classes=len(config.CLASSES)
    )
    
    model = compile_model(model, learning_rate=0.001)
    
    print("[INFO] Starting mini training (3 epochs)...")
    
    # Train for just a few epochs
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=32,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('demo_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("[INFO] Mini training completed!")
    print("[INFO] Training plot saved as 'demo_training.png'")
    
    # Make some predictions
    print("\n[INFO] Making sample predictions...")
    sample_predictions = model.predict(X_val[:5])
    
    for i, pred in enumerate(sample_predictions):
        top_class_idx = np.argmax(pred)
        confidence = pred[top_class_idx]
        predicted_class = config.CLASSES[top_class_idx]
        
        print(f"  Sample {i+1}: {predicted_class} (confidence: {confidence:.3f})")


def show_sample_images(X, num_samples=4):
    """
    Display sample synthetic X-ray images.
    
    Args:
        X: Array of images
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(12, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # Convert to grayscale for display
        img_gray = np.mean(X[i], axis=2)
        plt.imshow(img_gray, cmap='gray')
        plt.title(f'Sample {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Sample images saved as 'demo_samples.png'")


def main():
    """
    Run the complete demonstration.
    """
    print("="*60)
    print("NIH X-ray Neural Network - Demonstration")
    print("="*60)
    
    print("\n[INFO] This demo shows the capabilities of the NIH X-ray classification system")
    print("[INFO] using synthetic data (no real X-ray images required)")
    
    # Show available classes
    print(f"\n[INFO] Available pathology classes ({len(config.CLASSES)}):")
    for i, class_name in enumerate(config.CLASSES):
        print(f"  {i+1:2d}. {class_name}")
    
    # Demonstrate model architectures
    demonstrate_models()
    
    # Create and show sample synthetic data
    print("\n[INFO] Creating sample synthetic X-ray data...")
    X_sample, _, _, _ = create_synthetic_data(100)
    show_sample_images(X_sample)
    
    # Run mini training
    run_mini_training()
    
    print("\n" + "="*60)
    print("Demonstration completed!")
    print("="*60)
    
    print("\nTo use with real NIH data:")
    print("1. Download the NIH Chest X-ray dataset from Kaggle")
    print("2. Run: python scripts/prepare_dataset.py")
    print("3. Place dataset files in data/NIH/ directory")
    print("4. Run: python train.py --model custom --epochs 25")
    print("5. Evaluate: python evaluate.py")
    print("6. Predict: python predict.py --image path/to/xray.png")


if __name__ == "__main__":
    main()