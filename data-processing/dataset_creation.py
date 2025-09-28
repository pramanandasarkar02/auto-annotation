import os
import logging

from core import convert_dicom_to_nifti, convert_nrrd_to_nifti, image_label_shape_alignment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dicom_processing.log'),
        logging.StreamHandler()
    ]
)

def main():
    logger = logging.getLogger(__name__)
    root_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data"
    
    # convert dicom to nifti 
    ct_dir = os.path.join(root_dir, "ct")
    image_dir = os.path.join(root_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    convert_dicom_to_nifti(ct_dir, image_dir)

    # convert nrrd to nifti
    mask_dir = os.path.join(root_dir, "mask")
    label_dir = os.path.join(root_dir, "label")
    if not os.path.exists(mask_dir):
        logger.error(f"Mask directory {mask_dir} does not exist.")
        return
    os.makedirs(label_dir, exist_ok=True)
    convert_nrrd_to_nifti(mask_dir, label_dir)

    # alignment 
    image_dir = os.path.join(root_dir, "image")
    label_dir = os.path.join(root_dir, "label")
    image_label_shape_alignment(image_dir, label_dir)


if __name__ == "__main__":
    main()