# Dataset Setup Instructions

This project uses a local configuration approach to handle dataset paths, making it easy for different users to work with their own local copies of the NIH X-ray dataset.

## Quick Setup

### Option 1: Automatic Configuration (Recommended)
```bash
python scripts/prepare_dataset.py --setup-config
```
This will:
1. Prompt you for your local dataset paths
2. Create a `config/paths_local.py` file with your paths
3. Set up the dataset automatically
4. The paths file is ignored by git, so your local paths stay private

### Option 2: Manual Configuration
1. Copy `config/paths_template.py` to `config/paths_local.py`
2. Edit `config/paths_local.py` with your local paths:
   ```python
   IMAGES_PATH = "/your/path/to/nih/images"
   CSV_PATH = "/your/path/to/Data_Entry_2017.csv"
   COPY_FILES = False  # True to copy, False to create symbolic links
   ```
3. Run the setup:
   ```bash
   python scripts/prepare_dataset.py
   ```

### Option 3: Command Line (One-time)
```bash
python scripts/prepare_dataset.py --images-path /path/to/images --csv-path /path/to/csv
```

### Option 4: Interactive Mode
```bash
python scripts/prepare_dataset.py --interactive
```

## File Structure
After setup, your project will have:
```
data/
  NIH/
    images/          # Links to or copies of your NIH images
    Data_Entry_2017.csv  # Links to or copies of your CSV file
```

## Notes
- **Symbolic links** (default): Save disk space, requires original files to remain in place
- **Copy mode**: Uses more space but creates standalone copies
- The `config/paths_local.py` file is automatically ignored by git
- You can change your configuration anytime by running `--setup-config` again