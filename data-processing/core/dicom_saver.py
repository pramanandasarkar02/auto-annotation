import os
import uuid
import json
import logging
import SimpleITK as sitk
from .utils import DICOM_TAG_MAP
from .metadata_extractor import MetadataExtractor

class DicomSaver:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metadata_extractor = MetadataExtractor()

    def save_dicom_series(self, image, ref_files, output_dir, patient_id, phase, side, series_description):
        """Save DICOM series with preserved metadata"""
        phase_dir = os.path.join(output_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        
        series_uid = f"1.2.826.0.1.3680043.8.498.{uuid.uuid4().hex}"
        num_slices = image.GetDepth()
        
        self.logger.info(f"Saving {num_slices} slices for {patient_id} {phase} {side}")

        metadata_json = {
            "patient_id": patient_id,
            "phase": phase,
            "side": side,
            "series_description": series_description,
            "series_uid": series_uid,
            "total_slices": num_slices,
            "original_files_count": len(ref_files),
            "image_properties": self.metadata_extractor.extract_metadata(image),
            "slices": []
        }

        for i in range(num_slices):
            try:
                slice_img = image[:, :, i]
                slice_img = sitk.Cast(slice_img, sitk.sitkInt16)

                ref_index = i % len(ref_files)
                ref_reader = sitk.ImageFileReader()
                ref_reader.SetFileName(ref_files[ref_index])
                ref_reader.LoadPrivateTagsOn()
                ref_reader.ReadImageInformation()

                tags = {}
                for key in ref_reader.GetMetaDataKeys():
                    try:
                        value = ref_reader.GetMetaData(key)
                        slice_img.SetMetaData(key, value)
                        if key in DICOM_TAG_MAP:
                            tags[DICOM_TAG_MAP[key]] = value
                    except Exception as e:
                        self.logger.warning(f"Could not copy metadata key {key}: {e}")

                slice_img.SetMetaData("0020|000e", series_uid)
                slice_img.SetMetaData("0008|103e", series_description)
                slice_img.SetMetaData("0008|0018", f"1.2.826.0.1.3680043.8.498.{uuid.uuid4().hex}")
                slice_img.SetMetaData("0020|0013", str(i + 1))
                
                if ref_reader.HasMetaDataKey("0018|5100"):
                    slice_img.SetMetaData("0018|5100", ref_reader.GetMetaData("0018|5100"))
                if ref_reader.HasMetaDataKey("0020|0012"):
                    slice_img.SetMetaData("0020|0012", ref_reader.GetMetaData("0020|0012"))

                filename = f"{patient_id}_{phase}_{side}_{i+1:04d}.dcm"
                filepath = os.path.join(phase_dir, filename)
                
                writer.SetFileName(filepath)
                writer.Execute(slice_img)

                metadata_json["slices"].append({
                    "slice_number": i + 1,
                    "file": filename,
                    "reference_file": os.path.basename(ref_files[ref_index]),
                    "metadata": tags
                })
                
            except Exception as e:
                self.logger.error(f"Error saving slice {i}: {e}")
                continue

        metadata_file = os.path.join(phase_dir, f"metadata_{side}.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_json, f, indent=2)
        
        self.logger.info(f"Saved {len(metadata_json['slices'])} slices to {phase_dir}")
        return len(metadata_json['slices'])