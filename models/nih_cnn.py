"""
Neural Network model for NIH X-ray classification
Based on PyImageSearch tutorial adapted for medical image analysis
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class MedicalImageCNN:
    """
    Custom CNN for medical image classification
    """
    
    @staticmethod
    def build(width, height, depth, classes):
        """
        Build a custom CNN model for medical image classification.
        
        Args:
            width (int): Image width
            height (int): Image height
            depth (int): Number of channels
            classes (int): Number of output classes
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        model = Sequential()
        input_shape = (height, width, depth)
        
        # First CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third CONV => RELU => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Softmax classifier for multi-class classification
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model


class TransferLearningCNN:
    """
    Transfer learning model using pre-trained networks
    """
    
    @staticmethod
    def build_vgg16(width, height, depth, classes, freeze_base=True):
        """
        Build a transfer learning model using VGG16.
        
        Args:
            width (int): Image width
            height (int): Image height
            depth (int): Number of channels
            classes (int): Number of output classes
            freeze_base (bool): Whether to freeze base model weights
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Load the VGG16 network, ensuring the head FC layer sets are left off
        base_model = VGG16(
            weights="imagenet", 
            include_top=False,
            input_shape=(height, width, depth)
        )
        
        # Freeze the base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Construct the head of the model that will be placed on top of the base model
        head_model = base_model.output
        head_model = tf.keras.layers.GlobalAveragePooling2D()(head_model)
        head_model = Dense(512)(head_model)
        head_model = Activation("relu")(head_model)
        head_model = BatchNormalization()(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(classes)(head_model)
        head_model = Activation("softmax")(head_model)
        
        # Place the head FC model on top of the base model
        model = Model(inputs=base_model.input, outputs=head_model)
        
        return model
    
    @staticmethod
    def build_resnet50(width, height, depth, classes, freeze_base=True):
        """
        Build a transfer learning model using ResNet50.
        
        Args:
            width (int): Image width
            height (int): Image height
            depth (int): Number of channels
            classes (int): Number of output classes
            freeze_base (bool): Whether to freeze base model weights
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Load the ResNet50 network
        base_model = ResNet50(
            weights="imagenet", 
            include_top=False,
            input_shape=(height, width, depth)
        )
        
        # Freeze the base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Add custom head
        head_model = base_model.output
        head_model = tf.keras.layers.GlobalAveragePooling2D()(head_model)
        head_model = Dense(1024)(head_model)
        head_model = Activation("relu")(head_model)
        head_model = BatchNormalization()(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(512)(head_model)
        head_model = Activation("relu")(head_model)
        head_model = BatchNormalization()(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(classes)(head_model)
        head_model = Activation("softmax")(head_model)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=head_model)
        
        return model


def compile_model(model, learning_rate=1e-3, decay=1e-3):
    """
    Compile the model with appropriate optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate (float): Initial learning rate
        decay (float): Learning rate decay
        
    Returns:
        Compiled model
    """
    # Initialize the optimizer
    opt = Adam(learning_rate=learning_rate, decay=decay)
    
    # Compile the model
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=opt,
        metrics=["accuracy"]
    )
    
    return model