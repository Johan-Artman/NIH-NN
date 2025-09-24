"""
Utility script to download and prepare NIH X-ray dataset
"""

import os


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


def download_sample_data():
    """
    Download sample NIH dataset files for testing (if available).
    Note: The full NIH dataset is very large (45GB+) and should be downloaded manually.
    """
    print("[INFO] For the full NIH X-ray dataset, please visit:")
    print("https://www.kaggle.com/nih-chest-xrays/data")
    print("")
    print("The dataset contains:")
    print("- 112,120 X-ray images with disease labels")
    print("- Data_Entry_2017.csv with image labels")
    print("- 14 different pathology classes")
    print("")
    print("After downloading, extract the files to:")
    print("- data/NIH/images/ (for all .png image files)")
    print("- data/NIH/Data_Entry_2017.csv (labels file)")


if __name__ == "__main__":
    prepare_dataset_structure()
    download_sample_data()
    
    print("\n[INFO] Dataset preparation complete!")
    print("[INFO] Please download the NIH X-ray dataset manually and place files as instructed above.")