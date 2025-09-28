import os
import re
import logging
import SimpleITK as sitk
from core.utils import DICOM_TAG_MAP

class DicomLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_ct_phase_from_series(self, dicom_files):
        """Extract CT phase from series by analyzing multiple files and metadata"""
        phase_candidates = []
        
        # Try to extract from filenames
        for file_path in dicom_files[:5]:  # Check first 5 files
            basename = os.path.basename(file_path)
            patterns = [r"(CT\d+)", r"(\d+)", r"(Phase\d+)", r"(P\d+)"]
            for pattern in patterns:
                match = re.search(pattern, basename, re.IGNORECASE)
                if match:
                    phase_candidates.append(f"CT{match.group(1).replace('CT', '').replace('Phase', '').replace('P', '')}")
                    break
        
        # Try to extract from DICOM metadata
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(dicom_files[0])
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            if reader.HasMetaDataKey("0008|103e"):
                series_desc = reader.GetMetaData("0008|103e")
                patterns = [r"(CT\d+)", r"Phase\s*(\d+)", r"P(\d+)", r"(\d+)"]
                for pattern in patterns:
                    match = re.search(pattern, series_desc, re.IGNORECASE)
                    if match:
                        phase_candidates.append(f"CT{match.group(1).replace('CT', '')}")
                        break
            
            if reader.HasMetaDataKey("0020|000e"):
                series_uid = reader.GetMetaData("0020|000e")
                if not phase_candidates:
                    uid_suffix = series_uid.split('.')[-1][-3:]
                    try:
                        uid_num = int(uid_suffix) % 100
                        phase_candidates.append(f"CT{uid_num}")
                    except:
                        phase_candidates.append(f"CT{uid_suffix}")
                    
        except Exception as e:
            self.logger.warning(f"Could not read metadata from {dicom_files[0]}: {e}")
        
        if phase_candidates:
            return max(set(phase_candidates), key=phase_candidates.count)
        else:
            parent_dir = os.path.dirname(dicom_files[0])
            dir_match = re.search(r"(CT\d+)", parent_dir, re.IGNORECASE)
            if dir_match:
                return dir_match.group(1).upper()
            else:
                return f"CT_SERIES_{len(dicom_files)}"

    def load_dicom_series(self, patient_dir):
        """Load all DICOM series from patient directory"""
        reader = sitk.ImageSeriesReader()
        series_map = {}

        try:
            series_IDs = reader.GetGDCMSeriesIDs(patient_dir)
            self.logger.info(f"Found {len(series_IDs)} series IDs: {series_IDs}")
        except Exception as e:
            self.logger.error(f"Error getting series IDs: {e}")
            return {}

        for series_id in series_IDs:
            try:
                dicom_files = reader.GetGDCMSeriesFileNames(patient_dir, series_id)
                if dicom_files:
                    phase = self.get_ct_phase_from_series(dicom_files)
                    original_phase = phase
                    counter = 1
                    while phase in series_map:
                        phase = f"{original_phase}_{counter}"
                        counter += 1
                    series_map[phase] = dicom_files
                    self.logger.info(f"Series {series_id}: {len(dicom_files)} files -> {phase}")
                else:
                    self.logger.warning(f"No files found for series {series_id}")
            except Exception as e:
                self.logger.error(f"Error processing series {series_id}: {e}")
                continue

        return series_map