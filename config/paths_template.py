# NIH Dataset Paths Configuration
# Copy this file to 'paths_local.py' and update with your local paths
# The paths_local.py file is ignored by git so your local paths stay private

# Path to the directory containing NIH X-ray images (folder with .png files)
IMAGES_PATH = "/path/to/your/nih/images"

# Path to the Data_Entry_2017.csv file
CSV_PATH = "/path/to/your/Data_Entry_2017.csv"

# Whether to copy files (True) or create symbolic links (False)
# Symbolic links save disk space but require original files to remain in place
# Copying uses more space but creates a standalone copy
COPY_FILES = False