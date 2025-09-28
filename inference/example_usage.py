#!/usr/bin/env python3
"""
Example usage script for the Medical Image Inference CLI Tool
"""

import os
import sys
import tempfile
from pathlib import Path


def create_example_usage():
    """Show example usage scenarios"""

    print("Medical Image Inference CLI Tool - Examples")
    print("=" * 50)

    # Get the script directory
    script_dir = Path(__file__).parent

    print("\n1. BASIC USAGE:")
    print("=" * 30)
    print(f"python {script_dir}/main2.py \\")
    print("  --dcm_dir /path/to/patient_dicom_folder \\")
    print("  --out_dir /path/to/output_results \\")
    print("  --model_path /path/to/trained_model.pth")

    print("\n2. WITH LEFT/RIGHT SPLITTING:")
    print("=" * 30)
    print(f"python {script_dir}/main2.py \\")
    print("  --dcm_dir /path/to/patient_dicom_folder \\")
    print("  --out_dir /path/to/output_results \\")
    print("  --model_path /path/to/trained_model.pth \\")
    print("  --split \\")
    print("  --side left")

    print("\n3. CUSTOM MODEL NAME AND CPU PROCESSING:")
    print("=" * 30)
    print(f"python {script_dir}/main2.py \\")
    print("  --dcm_dir /path/to/patient_dicom_folder \\")
    print("  --out_dir /path/to/output_results \\")
    print("  --model_path /path/to/trained_model.pth \\")
    print("  --model_name FemurSegmentationUNet \\")
    print("  --device cpu \\")
    print("  --min_slices 15")

    print("\n4. PROCESS BOTH SIDES SEPARATELY:")
    print("=" * 30)
    print(f"python {script_dir}/main2.py \\")
    print("  --dcm_dir /path/to/patient_dicom_folder \\")
    print("  --out_dir /path/to/output_results \\")
    print("  --model_path /path/to/trained_model.pth \\")
    print("  --split \\")
    print("  --side both")

    print("\n" + "=" * 50)
    print("OUTPUT STRUCTURE:")
    print("=" * 50)
    print(
        """
output_dir/
├── {model_name}/
│   ├── {series_id}/
│   │   ├── {side}/
│   │   │   ├── dcm/           # DICOM outputs (future feature)
│   │   │   ├── nifti/         # NIfTI files
│   │   │   │   ├── prediction.nii.gz
│   │   │   │   └── image.nii.gz
│   │   │   ├── npy/           # NumPy arrays
│   │   │   │   ├── prediction.npy
│   │   │   │   └── image.npy
│   │   │   └── metadata.json  # Processing metadata
│   │   └── ...
│   └── ...
└── processing_summary.json    # Overall summary
    """
    )

    print("\nMETADATA INFORMATION:")
    print("=" * 50)
    print(
        """
The metadata.json contains:
- Processing timestamps and durations
- Image dimensions and properties
- Volume calculations for each class
- DICOM series information
- Configuration parameters used
    """
    )

    print("\nPREREQUISITES:")
    print("=" * 50)
    print(
        """
1. Install required packages:
   pip install torch torchvision monai nibabel SimpleITK numpy

2. Have a trained model file (.pth format)

3. DICOM files organized in a directory structure

4. Sufficient disk space for outputs
    """
    )


def show_help():
    """Show help information"""
    script_dir = Path(__file__).parent

    print("\nFor detailed help, run:")
    print(f"python {script_dir}/main2.py --help")

    print("\nFor simplified version:")
    print(f"python {script_dir}/main2_simple.py --help")


if __name__ == "__main__":
    create_example_usage()
    show_help()

    print("\n" + "=" * 50)
    print("Ready to process medical images!")
    print("=" * 50)
