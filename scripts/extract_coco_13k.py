import argparse
import json
import os
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import shutil
from tqdm import tqdm


def download_coco_annotations(output_dir: str) -> str:
    """Download COCO 2017 train annotations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    anno_zip = output_path / "annotations_trainval2017.zip"
    
    if not anno_zip.exists():
        print(f"Downloading COCO annotations from {anno_url}...")
        urlretrieve(anno_url, anno_zip)
        print(f"Downloaded to {anno_zip}")
        
        # Extract
        print("Extracting annotations...")
        with zipfile.ZipFile(anno_zip, 'r') as z:
            z.extractall(output_path)
        print("Annotations extracted.")
    
    return str(output_path / "annotations" / "instances_train2017.json")


def download_coco_images(anno_file: str, output_dir: str, num_images: int = 13000):
    """
    Download the first N COCO train images.
    
    Args:
        anno_file: Path to instances_train2017.json
        output_dir: Directory to save images
        num_images: Number of images to download (default 13000)
    """
    output_path = Path(output_dir) / "train2017"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {anno_file}...")
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    print(f"Total images in COCO train2017: {len(images)}")
    
    # Select first N images
    selected_images = images[:num_images]
    print(f"Selected first {len(selected_images)} images")
    
    # Download
    print(f"Downloading {len(selected_images)} images to {output_path}...")
    for img_info in tqdm(selected_images, desc="Downloading"):
        img_url = img_info['coco_url']
        img_filename = img_info['file_name']
        img_path = output_path / img_filename
        
        if img_path.exists():
            continue
        
        try:
            urlretrieve(img_url, img_path)
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
    
    print(f"Download complete. Images saved to {output_path}")
    return str(output_path)


def create_metadata(anno_file: str, output_dir: str, num_images: int = 13000):
    """Create a subset JSON with only the selected 13k images."""
    output_path = Path(output_dir)
    
    # Load full annotations
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    # Select first N images
    images = coco_data['images'][:num_images]
    image_ids = {img['id'] for img in images}
    
    # Filter annotations to only include selected images
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    
    # Create subset
    subset_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': images,
        'annotations': annotations
    }
    
    # Save
    subset_file = output_path / f"instances_train2017_{num_images}.json"
    with open(subset_file, 'w') as f:
        json.dump(subset_data, f)
    
    print(f"Saved metadata to {subset_file}")
    return str(subset_file)


def main():
    parser = argparse.ArgumentParser(description="Extract first 13,000 COCO train images")
    parser.add_argument('--output_dir', type=str, default='FINAL/data/data13k',
                        help='Output directory for images and annotations')
    parser.add_argument('--num_images', type=int, default=13000,
                        help='Number of images to extract')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    num_images = args.num_images
    
    print(f"=== Extracting first {num_images} COCO train images ===")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download annotations
    anno_file = download_coco_annotations(output_dir)
    
    # Step 2: Create metadata subset
    metadata_file = create_metadata(anno_file, output_dir, num_images)
    
    # Step 3: Download images
    images_dir = download_coco_images(anno_file, output_dir, num_images)
    
    print("\n=== Extraction complete ===")
    print(f"Images: {images_dir}")
    print(f"Metadata: {metadata_file}")


if __name__ == '__main__':
    main()
