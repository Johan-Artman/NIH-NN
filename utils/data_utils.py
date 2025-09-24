"""
Data preprocessing utilities for NIH X-ray dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset_metadata(csv_path):
    """
    Load and preprocess the NIH dataset metadata from CSV file.
    
    Args:
        csv_path (str): Path to the Data_Entry_2017.csv file
        
    Returns:
        pandas.DataFrame: Processed dataset metadata
    """
    print("[INFO] Loading dataset metadata...")
    df = pd.read_csv(csv_path)
    
    # Clean up the finding labels
    df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
    
    # Filter out images with no findings for balanced dataset (optional)
    # Uncomment the next line if you want to exclude "No Finding" cases
    # df = df[df['Finding Labels'].apply(lambda x: 'No Finding' not in x)]
    
    return df


def preprocess_labels(labels, classes):
    """
    Convert multi-label string format to binary matrix.
    
    Args:
        labels (list): List of label strings
        classes (list): List of all possible class names
        
    Returns:
        numpy.ndarray: Binary label matrix
    """
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    return mlb.transform(labels)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for the neural network.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {str(e)}")
        return None


def create_data_generators(train_df, val_df, image_dir, image_size, batch_size):
    """
    Create data generators for training and validation.
    
    Args:
        train_df (DataFrame): Training dataset metadata
        val_df (DataFrame): Validation dataset metadata
        image_dir (str): Directory containing images
        image_size (int): Size to resize images to
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=[0.9, 1.25],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='reflect'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=image_dir,
        x_col='Image Index',
        y_col='Finding Labels',
        class_mode='categorical',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=image_dir,
        x_col='Image Index',
        y_col='Finding Labels',
        class_mode='categorical',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator


def split_dataset(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        df (DataFrame): Dataset metadata
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=random_state,
        stratify=df['Finding Labels'].astype(str)  # Stratify by labels
    )
    
    # Second split: separate training and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df['Finding Labels'].astype(str)
    )
    
    return train_df, val_df, test_df