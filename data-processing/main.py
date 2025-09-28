import os
import json
import logging
import SimpleITK as sitk
from tqdm import tqdm

from core import DicomLoader, DicomSplitter, DicomSaver, DicomValidator

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
    raw_dir = os.path.join(root_dir, "raw")
    output_dir = os.path.join(root_dir, "ct")
    os.makedirs(output_dir, exist_ok=True)

    dicom_loader = DicomLoader()
    dicom_splitter = DicomSplitter()
    dicom_saver = DicomSaver()
    dicom_validator = DicomValidator()

    patient_ids = [p for p in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, p))]
    logger.info(f"Found {len(patient_ids)} patients to process")

    processing_summary = {
        "total_patients": len(patient_ids),
        "successful": 0,
        "failed": 0,
        "validation_passed": 0,
        "details": {}
    }

    for patient_id in tqdm(patient_ids, desc="Processing Patients"):
        logger.info(f"\n=== Processing Patient: {patient_id} ===")
        
        try:
            patient_path = os.path.join(raw_dir, patient_id)
            series_map = dicom_loader.load_dicom_series(patient_path)
            
            if not series_map:
                logger.warning(f"No series found for {patient_id}")
                processing_summary["failed"] += 1
                processing_summary["details"][patient_id] = "No series found"
                continue

            patient_results = {}
            total_saved = 0

            for phase, dicom_files in series_map.items():
                logger.info(f"\nProcessing phase: {phase} ({len(dicom_files)} files)")
                
                try:
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(dicom_files)
                    image = reader.Execute()
                    
                    logger.info(f"Loaded image: {image.GetSize()} voxels")

                    left_img, right_img = dicom_splitter.split_left_right(image)

                    left_out = os.path.join(output_dir, f"{patient_id}_left")
                    right_out = os.path.join(output_dir, f"{patient_id}_right")

                    left_saved = dicom_saver.save_dicom_series(
                        left_img, dicom_files, left_out, patient_id, phase, 
                        "left", f"{phase} Left Femur"
                    )
                    
                    right_saved = dicom_saver.save_dicom_series(
                        right_img, dicom_files, right_out, patient_id, phase, 
                        "right", f"{phase} Right Femur"
                    )
                    
                    patient_results[phase] = {
                        "original_files": len(dicom_files),
                        "left_saved": left_saved,
                        "right_saved": right_saved,
                        "total_saved": left_saved + right_saved
                    }
                    
                    total_saved += left_saved + right_saved
                    
                except Exception as e:
                    logger.error(f"Error processing phase {phase}: {e}")
                    patient_results[phase] = {"error": str(e)}
                    continue

            validation_passed = dicom_validator.validate_processing(raw_dir, output_dir, patient_id)
            
            processing_summary["successful"] += 1
            if validation_passed:
                processing_summary["validation_passed"] += 1
            
            processing_summary["details"][patient_id] = {
                "phases": patient_results,
                "total_saved": total_saved,
                "validation_passed": validation_passed
            }

        except Exception as e:
            logger.error(f"Error processing {patient_id}: {e}")
            processing_summary["failed"] += 1
            processing_summary["details"][patient_id] = {"error": str(e)}

    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(processing_summary, f, indent=2)

    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Total patients: {processing_summary['total_patients']}")
    logger.info(f"Successful: {processing_summary['successful']}")
    logger.info(f"Failed: {processing_summary['failed']}")
    logger.info(f"Validation passed: {processing_summary['validation_passed']}")
    logger.info(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()