"""
Training script for NIH X-ray classification
"""

import os
import argparse
import pickle
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.nih_cnn import MedicalImageCNN, TransferLearningCNN, compile_model
from utils.data_utils import load_dataset_metadata, split_dataset, create_data_generators


def plot_training_history(history, plot_path):
    """
    Plot training history and save to file.
    
    Args:
        history: Training history from model.fit()
        plot_path (str): Path to save the plot
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Training plot saved to {plot_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="custom", 
                       choices=["custom", "vgg16", "resnet50"],
                       help="Type of model to use")
    parser.add_argument("--dataset", type=str, default=config.DATASET_PATH,
                       help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                       help="Training batch size")
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(config.PLOT_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    
    print("[INFO] Loading and preprocessing dataset...")
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset path {args.dataset} does not exist!")
        print("[INFO] Please download the NIH X-ray dataset and place it in the data/ directory")
        print("[INFO] Dataset can be downloaded from: https://www.kaggle.com/nih-chest-xrays/data")
        return
    
    if not os.path.exists(config.LABELS_CSV):
        print(f"[ERROR] Labels CSV file {config.LABELS_CSV} does not exist!")
        print("[INFO] Please ensure Data_Entry_2017.csv is in the dataset directory")
        return
    
    # Load dataset metadata
    df = load_dataset_metadata(config.LABELS_CSV)
    print(f"[INFO] Loaded {len(df)} images from dataset")
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(
        df, 
        config.TRAIN_SPLIT, 
        config.VAL_SPLIT, 
        config.TEST_SPLIT
    )
    
    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Validation samples: {len(val_df)}")
    print(f"[INFO] Test samples: {len(test_df)}")
    
    # Initialize label binarizer
    lb = LabelBinarizer()
    all_labels = []
    for labels in df['Finding Labels']:
        all_labels.extend(labels)
    lb.fit(list(set(all_labels)))
    
    # Save label binarizer
    with open(config.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(lb, f)
    
    print(f"[INFO] Classes: {lb.classes_}")
    num_classes = len(lb.classes_)
    
    # Build model based on selection
    print(f"[INFO] Building {args.model} model...")
    
    if args.model == "custom":
        model = MedicalImageCNN.build(
            width=config.IMAGE_SIZE,
            height=config.IMAGE_SIZE,
            depth=3,  # RGB channels
            classes=num_classes
        )
    elif args.model == "vgg16":
        model = TransferLearningCNN.build_vgg16(
            width=config.IMAGE_SIZE,
            height=config.IMAGE_SIZE,
            depth=3,
            classes=num_classes
        )
    elif args.model == "resnet50":
        model = TransferLearningCNN.build_resnet50(
            width=config.IMAGE_SIZE,
            height=config.IMAGE_SIZE,
            depth=3,
            classes=num_classes
        )
    
    # Compile the model
    model = compile_model(model, config.INITIAL_LR, config.DECAY)
    
    print("[INFO] Model summary:")
    model.summary()
    
    # Create data generators
    image_dir = os.path.join(args.dataset, "images")
    if not os.path.exists(image_dir):
        print(f"[ERROR] Images directory {image_dir} does not exist!")
        print("[INFO] Please ensure images are in the data/NIH/images/ directory")
        return
    
    train_gen, val_gen = create_data_generators(
        train_df, val_df, image_dir, config.IMAGE_SIZE, args.batch_size
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("[INFO] Starting training...")
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_df) // args.batch_size
    validation_steps = len(val_df) // args.batch_size
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_path = os.path.join(config.PLOT_PATH, f"training_plot_{args.model}.png")
    plot_training_history(history, plot_path)
    
    print("[INFO] Training completed!")
    print(f"[INFO] Model saved to {config.MODEL_PATH}")
    print(f"[INFO] Training plot saved to {plot_path}")


if __name__ == "__main__":
    main()