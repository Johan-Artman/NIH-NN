"""
Utility script to download and prepare NIH X-ray dataset
"""

import os
import sys

# Add config directory to Python path
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
sys.path.insert(0, config_dir)


def load_paths_config():
    """
    Load paths configuration from config/paths_local.py
    Falls back to interactive input if config file doesn't exist.
    
    Returns:
        tuple: (images_path, csv_path, copy_files) or (None, None, None) if not found
    """
    try:
        import paths_local
        return paths_local.IMAGES_PATH, paths_local.CSV_PATH, getattr(paths_local, 'COPY_FILES', False)
    except ImportError:
        print("\n[INFO] Local paths configuration not found.")
        print("To set up automatic path loading:")
        print("1. Copy config/paths_template.py to config/paths_local.py")
        print("2. Edit config/paths_local.py with your local paths")
        print("3. Run this script again")
        return None, None, None


def create_paths_config_interactive():
    """
    Create a paths_local.py config file interactively.
    """
    print("\n[INFO] Creating local paths configuration...")
    
    images_path = input("Enter the path to your NIH X-ray images directory: ").strip()
    csv_path = input("Enter the path to your Data_Entry_2017.csv file: ").strip()
    
    while True:
        copy_choice = input("Copy files or create symbolic links? [C]opy/[L]ink: ").strip().upper()
        if copy_choice in ['C', 'L']:
            copy_files = copy_choice == 'C'
            break
        print("Please enter 'C' for copy or 'L' for link")
    
    # Verify paths exist
    if not os.path.exists(images_path):
        print(f"[WARNING] Images directory not found: {images_path}")
    
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}")
    
    config_content = f'''# NIH Dataset Paths Configuration
# This file is ignored by git so your local paths stay private

# Path to the directory containing NIH X-ray images (folder with .png files)
IMAGES_PATH = "{images_path}"

# Path to the Data_Entry_2017.csv file
CSV_PATH = "{csv_path}"

# Whether to copy files (True) or create symbolic links (False)
# Symbolic links save disk space but require original files to remain in place
# Copying uses more space but creates a standalone copy
COPY_FILES = {copy_files}
'''
    
    config_path = os.path.join(config_dir, 'paths_local.py')
    
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"[INFO] Configuration saved to: {config_path}")
        return images_path, csv_path, copy_files
    except Exception as e:
        print(f"[ERROR] Failed to save configuration: {e}")
        return None, None, None


def download_file(url, local_filename):
    """
    Download a file from URL with progress bar.
    
    Args:
        url (str): URL to download from
        local_filename (str): Local path to save the file
    """
    try:
        import requests
        from tqdm import tqdm
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except ImportError:
        print("[WARNING] requests and tqdm not available. Manual download required.")


def prepare_dataset_structure():
    """
    Create the required directory structure for the NIH dataset.
    """
    print("[INFO] Creating dataset directory structure...")
    
    base_dir = "data/NIH"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    
    print(f"[INFO] Created directory structure:")
    print(f"  - {base_dir}/")
    print(f"  - {base_dir}/images/")
    print(f"  - models/")
    print(f"  - plots/")
    print(f"  - evaluation_results/")


def copy_local_dataset(local_images_path, local_csv_path):
    """
    Copy NIH dataset files to local data directory (use for permanent setup).
    
    Args:
        local_images_path (str): Path to the directory containing NIH X-ray images
        local_csv_path (str): Path to the Data_Entry_2017.csv file
    """
    import shutil
    
    base_dir = "data/NIH"
    target_images_dir = os.path.join(base_dir, "images")
    target_csv_path = os.path.join(base_dir, "Data_Entry_2017.csv")
    
    print(f"[INFO] Copying dataset from local files...")
    print(f"[INFO] This may take a while for large datasets...")
    print(f"[INFO] Source images: {local_images_path}")
    print(f"[INFO] Source CSV: {local_csv_path}")
    
    # Verify source paths exist
    if not os.path.exists(local_images_path):
        print(f"[ERROR] Images directory not found: {local_images_path}")
        return False
    
    if not os.path.exists(local_csv_path):
        print(f"[ERROR] CSV file not found: {local_csv_path}")
        return False
    
    try:
        # Remove existing target directory if it exists
        if os.path.exists(target_images_dir):
            shutil.rmtree(target_images_dir)
        
        # Copy images directory
        print(f"[INFO] Copying images to {target_images_dir}...")
        shutil.copytree(local_images_path, target_images_dir)
        
        # Copy CSV file
        print(f"[INFO] Copying CSV to {target_csv_path}...")
        shutil.copy2(local_csv_path, target_csv_path)
        
        # Count images for verification
        image_count = len([f for f in os.listdir(target_images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"[INFO] Successfully copied {image_count} image files")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to copy files: {e}")
        return False


def link_local_dataset(local_images_path, local_csv_path):
    """
    Link NIH dataset files using symbolic links (saves disk space).
    
    Args:
        local_images_path (str): Path to the directory containing NIH X-ray images
        local_csv_path (str): Path to the Data_Entry_2017.csv file
    """
    base_dir = "data/NIH"
    target_images_dir = os.path.join(base_dir, "images")
    target_csv_path = os.path.join(base_dir, "Data_Entry_2017.csv")
    
    print(f"[INFO] Linking dataset from local files...")
    print(f"[INFO] Source images: {local_images_path}")
    print(f"[INFO] Source CSV: {local_csv_path}")
    
    # Verify source paths exist
    if not os.path.exists(local_images_path):
        print(f"[ERROR] Images directory not found: {local_images_path}")
        return False
    
    if not os.path.exists(local_csv_path):
        print(f"[ERROR] CSV file not found: {local_csv_path}")
        return False
    
    # Create symlinks to avoid copying large files
    try:
        # Remove existing target if it exists
        if os.path.exists(target_images_dir):
            if os.path.islink(target_images_dir):
                os.unlink(target_images_dir)
            elif os.path.isdir(target_images_dir):
                import shutil
                shutil.rmtree(target_images_dir)
        
        if os.path.exists(target_csv_path):
            if os.path.islink(target_csv_path):
                os.unlink(target_csv_path)
            elif os.path.isfile(target_csv_path):
                os.remove(target_csv_path)
        
        # Create symlinks
        os.symlink(os.path.abspath(local_images_path), target_images_dir)
        os.symlink(os.path.abspath(local_csv_path), target_csv_path)
        
        print(f"[INFO] Successfully linked images directory to: {target_images_dir}")
        print(f"[INFO] Successfully linked CSV file to: {target_csv_path}")
        
        # Count images for verification
        image_count = len([f for f in os.listdir(local_images_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"[INFO] Found {image_count} image files")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create symlinks: {e}")
        return False


def setup_local_dataset(local_images_path=None, local_csv_path=None):
    """
    Setup NIH dataset using locally available images and CSV file.
    Creates symbolic links by default to save disk space.
    
    Args:
        local_images_path (str): Path to the directory containing NIH X-ray images
        local_csv_path (str): Path to the Data_Entry_2017.csv file
    """
    base_dir = "data/NIH"
    target_images_dir = os.path.join(base_dir, "images")
    target_csv_path = os.path.join(base_dir, "Data_Entry_2017.csv")
    
    # If paths not provided, prompt user for them
    if local_images_path is None:
        local_images_path = input("Enter the path to your NIH X-ray images directory: ").strip()
    
    if local_csv_path is None:
        local_csv_path = input("Enter the path to your Data_Entry_2017.csv file: ").strip()
    
    # Ask user about linking vs copying
    while True:
        choice = input("\nChoose setup method:\n[L] Link (symbolic links - saves space, faster)\n[C] Copy (copies files - safer for permanent setup)\nChoice [L/C]: ").strip().upper()
        if choice in ['L', 'C']:
            break
        print("Please enter 'L' for link or 'C' for copy")
    
    if choice == 'C':
        return copy_local_dataset(local_images_path, local_csv_path)
    
    print(f"[INFO] Setting up dataset from local files...")
    print(f"[INFO] Source images: {local_images_path}")
    print(f"[INFO] Source CSV: {local_csv_path}")
    
    # Verify source paths exist
    if not os.path.exists(local_images_path):
        print(f"[ERROR] Images directory not found: {local_images_path}")
        return False
    
    if not os.path.exists(local_csv_path):
        print(f"[ERROR] CSV file not found: {local_csv_path}")
        return False
    
    # Create symlinks to avoid copying large files
    try:
        # Remove existing target if it exists
        if os.path.exists(target_images_dir):
            if os.path.islink(target_images_dir):
                os.unlink(target_images_dir)
            elif os.path.isdir(target_images_dir):
                import shutil
                shutil.rmtree(target_images_dir)
        
        if os.path.exists(target_csv_path):
            if os.path.islink(target_csv_path):
                os.unlink(target_csv_path)
            elif os.path.isfile(target_csv_path):
                os.remove(target_csv_path)
        
        # Create symlinks
        os.symlink(os.path.abspath(local_images_path), target_images_dir)
        os.symlink(os.path.abspath(local_csv_path), target_csv_path)
        
        print(f"[INFO] Successfully linked images directory to: {target_images_dir}")
        print(f"[INFO] Successfully linked CSV file to: {target_csv_path}")
        
        # Count images for verification
        image_count = len([f for f in os.listdir(local_images_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"[INFO] Found {image_count} image files")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create symlinks: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare NIH X-ray dataset')
    parser.add_argument('--images-path', type=str, 
                       help='Path to directory containing NIH X-ray images')
    parser.add_argument('--csv-path', type=str,
                       help='Path to Data_Entry_2017.csv file')
    parser.add_argument('--copy', action='store_true',
                       help='Copy files instead of creating symbolic links (requires more disk space)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode - prompt for paths and options')
    parser.add_argument('--setup-config', action='store_true',
                       help='Create a local paths configuration file')
    
    args = parser.parse_args()
    
    # Always prepare directory structure first
    prepare_dataset_structure()
    
    # Handle setup-config mode
    if args.setup_config:
        images_path, csv_path, copy_files = create_paths_config_interactive()
        if images_path and csv_path:
            print("\n[INFO] Configuration created! You can now run the script without arguments.")
            # Continue with setup using the new config
        else:
            print("\n[ERROR] Failed to create configuration")
            sys.exit(1)
    elif args.images_path and args.csv_path:
        # Use provided command-line paths
        images_path = args.images_path
        csv_path = args.csv_path
        copy_files = args.copy
    elif args.interactive:
        # Interactive mode
        success = setup_local_dataset()
        if success:
            print("\n[INFO] Local dataset setup complete!")
        else:
            print("\n[ERROR] Failed to setup local dataset")
        sys.exit(0)
    else:
        # Try to load from config file
        images_path, csv_path, copy_files = load_paths_config()
        
        if not images_path or not csv_path:
            print("\n[ERROR] No paths provided and no configuration file found.")
            print("\nOptions:")
            print("1. Use command line: --images-path /path/to/images --csv-path /path/to/csv")
            print("2. Create config file: --setup-config")
            print("3. Interactive mode: --interactive")
            print("\nExample:")
            print("  python scripts/prepare_dataset.py --setup-config")
            sys.exit(1)
    
    # Setup dataset using determined paths and method
    if copy_files:
        success = copy_local_dataset(images_path, csv_path)
    else:
        success = link_local_dataset(images_path, csv_path)
    
    if success:
        print("\n[INFO] Local dataset setup complete!")
        print(f"[INFO] Method: {'Copy' if copy_files else 'Symbolic links'}")
        print(f"[INFO] Images: {images_path}")
        print(f"[INFO] CSV: {csv_path}")
    else:
        print("\n[ERROR] Failed to setup local dataset")