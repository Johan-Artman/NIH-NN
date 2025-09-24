"""
Evaluation script for NIH X-ray classification model
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.data_utils import load_dataset_metadata, split_dataset, load_and_preprocess_image


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {output_path}")


def plot_roc_curves(y_true, y_pred_proba, classes, output_path):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        classes: Class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-Class Classification')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curves saved to {output_path}")


def evaluate_model(model, test_data, test_labels, lb, output_dir):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test images
        test_labels: True test labels
        lb: Label binarizer
        output_dir: Directory to save evaluation results
    """
    print("[INFO] Evaluating model on test data...")
    
    # Make predictions
    predictions = model.predict(test_data, batch_size=config.BATCH_SIZE)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    # Generate classification report
    report = classification_report(
        true_labels, pred_labels, 
        target_names=lb.classes_, 
        output_dict=True
    )
    
    print("\n[INFO] Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=lb.classes_))
    
    # Save detailed report
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(true_labels, pred_labels, target_names=lb.classes_))
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(true_labels, pred_labels, lb.classes_, cm_path)
    
    # Plot ROC curves
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plot_roc_curves(test_labels, predictions, lb.classes_, roc_path)
    
    # Calculate and display accuracy
    accuracy = (pred_labels == true_labels).mean()
    print(f"\n[INFO] Test Accuracy: {accuracy:.4f}")
    
    return report, accuracy


def load_test_data(test_df, image_dir, lb):
    """
    Load and preprocess test data.
    
    Args:
        test_df: Test dataset DataFrame
        image_dir: Directory containing images
        lb: Label binarizer
        
    Returns:
        tuple: (test_images, test_labels)
    """
    print("[INFO] Loading test images...")
    
    test_images = []
    test_labels = []
    
    for idx, row in test_df.iterrows():
        image_path = os.path.join(image_dir, row['Image Index'])
        image = load_and_preprocess_image(image_path, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        if image is not None:
            test_images.append(image)
            # Convert labels to binary format
            labels = lb.transform([row['Finding Labels']])
            test_labels.append(labels[0])
        
        if len(test_images) % 100 == 0:
            print(f"[INFO] Loaded {len(test_images)} test images...")
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"[INFO] Loaded {len(test_images)} test images")
    return test_images, test_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.MODEL_PATH,
                       help="Path to trained model")
    parser.add_argument("--dataset", type=str, default=config.DATASET_PATH,
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="evaluation_results/",
                       help="Output directory for evaluation results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file {args.model} does not exist!")
        print("[INFO] Please train the model first using train.py")
        return
    
    # Load the trained model
    print(f"[INFO] Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    
    # Load label binarizer
    with open(config.LABEL_ENCODER_PATH, "rb") as f:
        lb = pickle.load(f)
    
    print(f"[INFO] Classes: {lb.classes_}")
    
    # Load dataset metadata
    df = load_dataset_metadata(config.LABELS_CSV)
    
    # Split dataset to get test set
    train_df, val_df, test_df = split_dataset(
        df, 
        config.TRAIN_SPLIT, 
        config.VAL_SPLIT, 
        config.TEST_SPLIT
    )
    
    print(f"[INFO] Test samples: {len(test_df)}")
    
    # Load test data
    image_dir = os.path.join(args.dataset, "images")
    test_images, test_labels = load_test_data(test_df, image_dir, lb)
    
    # Evaluate the model
    report, accuracy = evaluate_model(model, test_images, test_labels, lb, args.output)
    
    print(f"\n[INFO] Evaluation completed!")
    print(f"[INFO] Results saved to {args.output}")
    print(f"[INFO] Final Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()