# NIH X-ray Neural Network

A deep learning project for medical image analysis using the NIH Chest X-ray dataset. This implementation is based on the PyImageSearch tutorial for medical image analysis with Keras, adapted specifically for the NIH X-ray dataset.

## Overview

This project implements a convolutional neural network (CNN) for classifying chest X-ray images into 15 different categories including 14 pathology types and normal cases. The model can help in automated screening and diagnosis of chest-related diseases.

### Supported Pathologies
- Atelectasis
- Cardiomegaly  
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia
- No Finding (Normal)

## Features

- **Multiple Model Architectures**: Custom CNN, VGG16, and ResNet50 with transfer learning
- **Data Augmentation**: Built-in image augmentation for better generalization
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and ROC curves
- **Batch Prediction**: Process multiple images at once
- **Visualization**: Generate prediction visualizations with confidence scores
- **Modular Design**: Easy to extend and customize

## Project Structure

```
NIH-NN/
├── config/
│   └── config.py              # Configuration settings
├── data/
│   └── NIH/                   # Dataset directory (to be populated)
│       ├── images/            # X-ray image files
│       └── Data_Entry_2017.csv # Labels file
├── models/
│   ├── nih_cnn.py            # Model architectures
│   ├── nih_xray_model.h5     # Trained model (generated)
│   └── label_encoder.pickle   # Label encoder (generated)
├── utils/
│   ├── __init__.py
│   └── data_utils.py         # Data preprocessing utilities
├── scripts/
│   └── prepare_dataset.py    # Dataset preparation script
├── plots/                    # Training plots (generated)
├── evaluation_results/       # Evaluation outputs (generated)
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── predict.py                # Prediction script
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Johan-Artman/NIH-NN.git
cd NIH-NN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare dataset structure:
```bash
python scripts/prepare_dataset.py
```

## Dataset Setup

1. Download the NIH Chest X-ray dataset from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data)
2. Extract the dataset and organize as follows:
   - Place all image files in `data/NIH/images/`
   - Place `Data_Entry_2017.csv` in `data/NIH/`

The dataset contains 112,120 frontal-view X-ray images of 30,805 unique patients with 14 disease labels.

## Usage

### Training

Train a model using one of the available architectures:

```bash
# Train custom CNN
python train.py --model custom --epochs 25 --batch-size 32

# Train with VGG16 transfer learning
python train.py --model vgg16 --epochs 25 --batch-size 16

# Train with ResNet50 transfer learning  
python train.py --model resnet50 --epochs 25 --batch-size 16
```

### Evaluation

Evaluate the trained model on test data:

```bash
python evaluate.py --model models/nih_xray_model.h5 --output evaluation_results/
```

This generates:
- Classification report
- Confusion matrix
- ROC curves
- Detailed performance metrics

### Prediction

Make predictions on new X-ray images:

```bash
# Single image prediction
python predict.py --image path/to/xray.png --visualize

# Batch prediction
python predict.py --image-dir path/to/images/ --output predictions.csv
```

## Configuration

Modify `config/config.py` to adjust:
- Image size and batch size
- Learning rates and training epochs
- Dataset paths
- Model architecture parameters

## Model Performance

The models achieve competitive performance on the NIH dataset:
- Custom CNN: Baseline performance with fast training
- VGG16: Improved accuracy with transfer learning
- ResNet50: Best performance for complex pathology detection

## Key Components

### Data Preprocessing
- Image resizing and normalization
- Multi-label encoding for pathologies
- Train/validation/test splitting
- Data augmentation for training robustness

### Model Architectures
- **Custom CNN**: 3-layer CNN with batch normalization and dropout
- **VGG16**: Transfer learning with custom classification head
- **ResNet50**: Deep residual network with fine-tuning capability

### Training Features
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- Training progress visualization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Based on the PyImageSearch tutorial for medical image analysis
- NIH Clinical Center for providing the chest X-ray dataset
- TensorFlow and Keras communities for the deep learning framework

## Citation

If you use this code in your research, please consider citing:

```
Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). 
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks for 
Weakly-Supervised Classification and Localization of Common Thorax Diseases. 
IEEE CVPR 2017.
```