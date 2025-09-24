"""
Prediction script for NIH X-ray classification
"""

import os
import argparse
import pickle
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.data_utils import load_and_preprocess_image


def predict_image(model, image_path, lb, top_k=5):
    """
    Predict pathologies for a single X-ray image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        lb: Label binarizer
        top_k: Number of top predictions to return
        
    Returns:
        list: Top-k predictions with confidence scores
    """
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        class_name = lb.classes_[i]
        confidence = predictions[i]
        results.append((class_name, confidence))
    
    return results


def predict_batch(model, image_dir, lb, output_file=None, top_k=3):
    """
    Predict pathologies for multiple X-ray images.
    
    Args:
        model: Trained Keras model
        image_dir: Directory containing images
        lb: Label binarizer
        output_file: Optional file to save results
        top_k: Number of top predictions per image
    """
    print(f"[INFO] Processing images in {image_dir}...")
    
    results = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        predictions = predict_image(model, image_path, lb, top_k)
        
        if predictions is not None:
            results[filename] = predictions
            
            print(f"\n[INFO] Image: {filename}")
            for j, (class_name, confidence) in enumerate(predictions):
                print(f"  {j+1}. {class_name}: {confidence:.4f}")
        
        if (i + 1) % 10 == 0:
            print(f"[INFO] Processed {i+1}/{len(image_files)} images")
    
    # Save results to file if specified
    if output_file:
        print(f"[INFO] Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            f.write("Image,Top_Predictions\n")
            for filename, predictions in results.items():
                pred_str = " | ".join([f"{cls}:{conf:.4f}" for cls, conf in predictions])
                f.write(f"{filename},{pred_str}\n")
    
    return results


def visualize_prediction(model, image_path, lb, output_path=None, top_k=5):
    """
    Visualize prediction results on an X-ray image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        lb: Label binarizer
        output_path: Path to save the visualization
        top_k: Number of top predictions to show
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib is required for visualization")
        return
    
    # Load original image for display
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    predictions = predict_image(model, image_path, lb, top_k)
    
    if predictions is None:
        return
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"X-ray Image\n{os.path.basename(image_path)}")
    plt.axis('off')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    classes = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]
    
    y_pos = np.arange(len(classes))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    bars = plt.barh(y_pos, confidences, color=colors)
    plt.yticks(y_pos, classes)
    plt.xlabel('Confidence Score')
    plt.title(f'Top {top_k} Predictions')
    plt.xlim(0, 1)
    
    # Add confidence values on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{conf:.3f}', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.MODEL_PATH,
                       help="Path to trained model")
    parser.add_argument("--image", type=str, required=False,
                       help="Path to single image for prediction")
    parser.add_argument("--image-dir", type=str, required=False,
                       help="Directory containing images for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Output file for batch predictions")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization of predictions")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top predictions to show")
    args = parser.parse_args()
    
    # Check arguments
    if not args.image and not args.image_dir:
        print("[ERROR] Please specify either --image or --image-dir")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file {args.model} does not exist!")
        print("[INFO] Please train the model first using train.py")
        return
    
    # Load the trained model
    print(f"[INFO] Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Load label binarizer
    with open(config.LABEL_ENCODER_PATH, "rb") as f:
        lb = pickle.load(f)
    
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Available classes: {', '.join(lb.classes_)}")
    
    if args.image:
        # Single image prediction
        print(f"[INFO] Predicting for image: {args.image}")
        predictions = predict_image(model, args.image, lb, args.top_k)
        
        if predictions:
            print(f"\n[INFO] Top {args.top_k} predictions:")
            for i, (class_name, confidence) in enumerate(predictions):
                print(f"  {i+1}. {class_name}: {confidence:.4f}")
            
            # Create visualization if requested
            if args.visualize:
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                vis_path = f"{base_name}_prediction.png"
                visualize_prediction(model, args.image, lb, vis_path, args.top_k)
    
    elif args.image_dir:
        # Batch prediction
        print(f"[INFO] Processing images in directory: {args.image_dir}")
        results = predict_batch(model, args.image_dir, lb, args.output, args.top_k)
        print(f"\n[INFO] Processed {len(results)} images")
        print(f"[INFO] Results saved to {args.output}")


if __name__ == "__main__":
    main()