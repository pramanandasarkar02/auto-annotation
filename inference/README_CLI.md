# Medical Image Inference CLI Tool

This command-line tool provides automated inference for medical image segmentation using trained UNet3D models. It processes DICOM files and generates predictions with comprehensive output structure and metadata.

## Features

- **DICOM Processing**: Automatically loads and processes DICOM series
- **Model Inference**: Runs inference using trained UNet3D models
- **Left/Right Splitting**: Optional anatomical splitting functionality
- **Multiple Output Formats**: Saves results as DICOM, NIfTI, and NumPy arrays
- **Comprehensive Metadata**: Tracks processing time, volumes, and metrics
- **Flexible Configuration**: Command-line arguments for customization

## Output Structure

The tool creates the following directory structure:

```
output_dir/
├── {model_name}/
│   ├── {series_id}/
│   │   ├── {side}/
│   │   │   ├── dcm/           # DICOM outputs
│   │   │   ├── nifti/         # NIfTI files (prediction.nii.gz, image.nii.gz)
│   │   │   ├── npy/           # NumPy arrays (prediction.npy, image.npy)
│   │   │   └── metadata.json  # Processing metadata
│   │   └── ...
│   └── ...
└── processing_summary.json    # Overall processing summary
```

## Usage

### Basic Usage

```bash
python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth
```

### Advanced Usage

```bash
# Process with left/right splitting
python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth --split --side left

# Custom model name and minimum slices
python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth --model_name CustomUNet --min_slices 15

# Process both sides separately
python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth --split --side both
```

## Command Line Arguments

### Required Arguments

- `--dcm_dir`: Path to DICOM directory containing the medical images
- `--out_dir`: Output directory for saving results
- `--model_path`: Path to the trained model file (.pth format)

### Optional Arguments

- `--split`: Enable left/right anatomical splitting (default: False)
- `--model_name`: Model name for output directory structure (default: 'UNet3D')
- `--min_slices`: Minimum number of slices required for processing (default: 10)
- `--side`: Which side to process when splitting - choices: 'left', 'right', 'both' (default: 'both')
- `--device`: Device for inference - 'cuda' or 'cpu' (default: 'cuda')

## Metadata Information

The `metadata.json` file contains:

- **Processing Information**: Timestamps, processing times, inference duration
- **Image Properties**: Original and processed shapes, spacing, orientation
- **Volume Metrics**: Voxel counts and volumes for each segmented class
- **DICOM Details**: Series information, file counts
- **Configuration**: Processing parameters used

## Example Metadata

```json
{
  "series_id": "1.2.826.0.1.3680043.8.498.123456789",
  "side": "right",
  "processing_timestamp": "2025-08-29T10:30:00",
  "total_processing_time": 45.67,
  "inference_time": 12.34,
  "original_shape": [512, 512, 100],
  "processed_shape": [256, 256, 100],
  "total_slices": 100,
  "volume_metrics": {
    "class_0_volume": 1234567.89,
    "class_1_volume": 98765.43,
    "class_2_volume": 45678.9,
    "class_3_volume": 12345.67
  },
  "unique_classes": [0, 1, 2, 3]
}
```

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision
pip install monai
pip install nibabel
pip install SimpleITK
pip install numpy
pip install scipy
```

Or install from the project requirements:

```bash
pip install -r requirements.txt
```

## Error Handling

The tool includes comprehensive error handling:

- **Missing Files**: Validates existence of DICOM directory and model file
- **Insufficient Data**: Checks minimum slice requirements
- **CUDA Availability**: Automatically falls back to CPU if CUDA is unavailable
- **DICOM Validation**: Skips corrupted or invalid DICOM files
- **Processing Errors**: Continues processing other series if one fails

## Examples

### Example 1: Basic Femur Segmentation

```bash
python main2.py \
  --dcm_dir /data/patient_001/CT_series \
  --out_dir /results/patient_001 \
  --model_path /models/femur_unet3d.pth
```

### Example 2: Left Femur Only with Custom Settings

```bash
python main2.py \
  --dcm_dir /data/patient_002/CT_series \
  --out_dir /results/patient_002 \
  --model_path /models/femur_unet3d.pth \
  --split \
  --side left \
  --model_name FemurUNet3D \
  --min_slices 20
```

### Example 3: Batch Processing with CPU

```bash
python main2.py \
  --dcm_dir /data/batch_processing \
  --out_dir /results/batch_output \
  --model_path /models/bone_segmentation.pth \
  --device cpu \
  --min_slices 5
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use `--device cpu` or reduce batch size in the code
2. **Import Errors**: Ensure all dependencies are installed and paths are correct
3. **DICOM Loading Errors**: Check DICOM file integrity and format
4. **Permission Errors**: Ensure write permissions for output directory

### Logs

The tool provides detailed logging information. Look for:

- Series detection and validation messages
- Processing progress updates
- Error messages with specific failure reasons
- Performance metrics (processing and inference times)

## Integration

This tool can be integrated into larger processing pipelines:

```python
from inference.main2 import InferenceCLI

cli = InferenceCLI()
model = cli.load_model('path/to/model.pth')
# Process programmatically...
```

## Support

For issues or questions:

1. Check the logs for detailed error information
2. Verify all dependencies are correctly installed
3. Ensure DICOM files are valid and accessible
4. Check model file format and compatibility
