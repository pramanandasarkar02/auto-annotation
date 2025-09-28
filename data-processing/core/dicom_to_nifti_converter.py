
import os
import SimpleITK as sitk
from tqdm import tqdm
import logging

def convert_dicom_series(dicom_dir, output_path):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)

    if not dicom_files:
        logging.warning(f"No DICOM files found in {dicom_dir}")
        return 

    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    print(image.GetSize())
    sitk.WriteImage(image, output_path)
    
    logging.info(f"Converted{os.path.basename(output_path)}")


def convert_dicom_to_nifti(patient_dir,image_dir, min_dcm_images=10):
    for patient in os.listdir(patient_dir):
        patient_path = os.path.join(patient_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        for series in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series)
            if not os.path.isdir(series_path):
                logging.warning(f"Skipping non-directory file: {series_path}")
                continue
            if len(os.listdir(series_path)) < min_dcm_images:
                logging.warning(f"Skipping series with less than {min_dcm_images} DICOM files: {series_path}")
                continue

            
            dicom_dir = series_path
            output_dir = os.path.join(image_dir, series)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{patient}_{series}.nii.gz")
            convert_dicom_series(dicom_dir, output_path)