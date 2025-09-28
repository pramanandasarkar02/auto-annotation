import os
import logging
from typing import List, Tuple

def get_data_list(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Create a list of (image_path, label_path) pairs from directory structure.
    
    Args:
        images_dir: Directory containing image series folders
        labels_dir: Directory containing label series folders
    
    Returns:
        List of tuples containing (image_path, label_path)
    """
    data_list = []
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        logging.error(f"Images dir {images_dir} or labels dir {labels_dir} does not exist")
        return data_list
    
    for series_id in os.listdir(images_dir):
        image_series_path = os.path.join(images_dir, series_id)
        label_series_path = os.path.join(labels_dir, series_id)

        if not os.path.isdir(image_series_path) or not os.path.isdir(label_series_path):
            continue

        for image_id in os.listdir(image_series_path):
            image_path = os.path.join(image_series_path, image_id)

            # Get patient prefix without extension
            base_name = os.path.splitext(os.path.splitext(image_id)[0])[0]

            # Try to find matching label file in label_series_path
            matches = [f for f in os.listdir(label_series_path) if base_name in f]

            if not matches:
                logging.warning(f"No label found for {image_id}")
                continue

            label_path = os.path.join(label_series_path, matches[0])
            data_list.append({"image": image_path, "label": label_path})
    
    return data_list