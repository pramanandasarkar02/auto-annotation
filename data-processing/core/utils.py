# DICOM tag map to readable metadata keys
DICOM_TAG_MAP = {
    "0008|0020": "StudyDate",
    "0008|0060": "Modality",
    "0010|0010": "PatientName",
    "0010|0020": "PatientID",
    "0018|0050": "SliceThickness",
    "0028|0030": "PixelSpacing",
    "0028|0010": "Rows",
    "0028|0011": "Columns",
    "0028|0100": "BitsAllocated",
    "0028|1050": "WindowCenter",
    "0028|1051": "WindowWidth",
    "0020|0032": "ImagePositionPatient",
    "0020|0037": "ImageOrientationPatient",
    "0018|0060": "KVP",
    "0008|103e": "SeriesDescription",
    "0020|000e": "SeriesInstanceUID"
}