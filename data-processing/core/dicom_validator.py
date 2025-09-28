import os
import logging

class DicomValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_processing(self, raw_dir, output_dir, patient_id):
        """Validate that processing preserved correct number of images"""
        raw_path = os.path.join(raw_dir, patient_id)
        
        original_count = 0
        for root, _, files in os.walk(raw_path):
            dicom_files = [f for f in files if (
                f.lower().endswith('.dcm') or 
                f.lower().endswith('.dicom') or
                (not '.' in f and len(f) > 5)
            )]
            original_count += len(dicom_files)
        
        processed_count = 0
        if os.path.exists(output_dir):
            patient_dirs = [d for d in os.listdir(output_dir) if d.startswith(patient_id)]
            for patient_dir in patient_dirs:
                patient_path = os.path.join(output_dir, patient_dir)
                if os.path.exists(patient_path):
                    for root, _, files in os.walk(patient_path):
                        processed_count += len([f for f in files if f.lower().endswith('.dcm')])
        
        self.logger.info(f"Validation for {patient_id}:")
        self.logger.info(f"  Original DICOM files: {original_count}")
        self.logger.info(f"  Processed DICOM files: {processed_count}")
        self.logger.info(f"  Expected after split: {original_count * 2}")
        self.logger.info(f"  Match: {'✓' if processed_count == original_count * 2 else '✗'}")
        
        return processed_count == original_count * 2