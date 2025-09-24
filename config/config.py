# Configuration for NIH X-ray Dataset Neural Network

# Dataset configuration
DATASET_PATH = "data/NIH/"
LABELS_CSV = "data/NIH/Data_Entry_2017.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Training configuration
EPOCHS = 25
INITIAL_LR = 1e-3
DECAY = 1e-3 / EPOCHS

# Model configuration
MODEL_PATH = "models/nih_xray_model.h5"
PLOT_PATH = "plots/"
LABEL_ENCODER_PATH = "models/label_encoder.pickle"

# Classes in NIH dataset (14 pathology classes + 1 no finding)
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

# Data split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1