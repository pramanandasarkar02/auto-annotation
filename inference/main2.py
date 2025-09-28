import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# is it possible to run inference with out creating model 

def create_unet3d_model(in_channels: int = 1, out_channels: int = 4):
    """Create a 3D UNet model for medical image segmentation"""
    try:
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    except ImportError:
        logger.error("MONAI not installed. Please install it: pip install monai")
        sys.exit(1)


class InferenceCLI:
    """Command Line Interface for Medical Image Inference"""

    def __init__(self):
        # Initialize without external dependencies
        pass

    def create_output_structure(
        self, output_dir: str, model_name: str, series_id: str, side: str
    ) -> Dict[str, str]:
        """Create the required output directory structure"""
        base_path = Path(output_dir) / model_name / series_id / side

        paths = {
            "base": str(base_path),
            "dcm": str(base_path / "dcm"),
            "nifti": str(base_path / "nifti"),
            "npy": str(base_path / "npy"),
            "metadata": str(base_path / "metadata.json"),
        }

        # Create directories
        for key in ["dcm", "nifti", "npy"]:
            os.makedirs(paths[key], exist_ok=True)

        return paths

    def load_model(self, model_path: str, device: str = "cuda") -> torch.nn.Module:
        """Load the trained model"""
        logger.info(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create model
        model = create_unet3d_model(in_channels=1, out_channels=4)

        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
        return model

    def process_dicom_directory(
        self, dcm_dir: str, min_slices: int = 10
    ) -> List[Tuple[str, List[str]]]:
        """Process DICOM directory and return series information"""
        logger.info(f"Processing DICOM directory: {dcm_dir}")

        if not os.path.exists(dcm_dir):
            raise FileNotFoundError(f"DICOM directory not found: {dcm_dir}")

        # Get all DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(dcm_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Check if it's a DICOM file
                if file.lower().endswith(".dcm") or self._is_dicom_file(file_path):
                    dicom_files.append(file_path)

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dcm_dir}")

        # Group by series
        series_groups: Dict[str, List[str]] = {}
        for dcm_file in dicom_files:
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(dcm_file)
                reader.ReadImageInformation()

                series_uid = (
                    reader.GetMetaData("0020|000e")
                    if reader.HasMetaDataKey("0020|000e")
                    else "unknown"
                )

                if series_uid not in series_groups:
                    series_groups[series_uid] = []
                series_groups[series_uid].append(dcm_file)

            except Exception as e:
                logger.warning(f"Failed to read DICOM file {dcm_file}: {e}")
                continue

        # Filter by minimum slices
        valid_series = []
        for series_uid, files in series_groups.items():
            if len(files) >= min_slices:
                valid_series.append((series_uid, sorted(files)))
                logger.info(f"Found series {series_uid} with {len(files)} slices")
            else:
                logger.warning(
                    f"Series {series_uid} has only {len(files)} slices (< {min_slices}), skipping"
                )

        return valid_series

    def _is_dicom_file(self, file_path: str) -> bool:
        """Check if a file is a DICOM file"""
        try:
            with open(file_path, "rb") as f:
                # DICOM files start with a 128-byte preamble followed by 'DICM'
                f.seek(128)
                return f.read(4) == b"DICM"
        except:
            return False

    def split_left_right(
        self, image_data: np.ndarray, split_method: str = "center"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split image into left and right parts"""
        if split_method == "center":
            mid_point = image_data.shape[0] // 2
            left_part = image_data[:mid_point, :, :]
            right_part = image_data[mid_point:, :, :]
        else:
            # Could implement more sophisticated splitting methods here
            raise NotImplementedError(f"Split method '{split_method}' not implemented")

        return left_part, right_part

    def run_inference_on_image(
        self, model: torch.nn.Module, image_data: np.ndarray, device: str = "cuda"
    ) -> np.ndarray:
        """Run inference on image data"""
        logger.info("Running inference...")
        start_time = time.time()

        # Convert to tensor and add batch/channel dimensions
        image_tensor = (
            torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # Run inference
        with torch.no_grad():
            from monai.inferers import sliding_window_inference

            pred_logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
            pred_labels = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        return pred_labels, inference_time

    def calculate_volume_metrics(
        self, prediction: np.ndarray, spacing: Tuple[float, float, float]
    ) -> Dict:
        """Calculate volume metrics for each class"""
        voxel_volume = np.prod(spacing)
        volumes = {}

        unique_classes = np.unique(prediction)
        for class_id in unique_classes:
            voxel_count = np.sum(prediction == class_id)
            volume_mm3 = voxel_count * voxel_volume
            volumes[f"class_{int(class_id)}_volume"] = volume_mm3
            volumes[f"class_{int(class_id)}_voxels"] = int(voxel_count)

        return volumes

    def save_results(
        self,
        paths: Dict[str, str],
        image_data: np.ndarray,
        prediction: np.ndarray,
        affine: np.ndarray,
        spacing: Tuple[float, float, float],
        metadata: Dict,
    ):
        """Save all results to appropriate formats"""

        # Save NIfTI
        nifti_path = os.path.join(paths["nifti"], "prediction.nii.gz")
        nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), nifti_path)
        logger.info(f"Saved NIfTI prediction: {nifti_path}")

        # Save original image as NIfTI for reference
        image_nifti_path = os.path.join(paths["nifti"], "image.nii.gz")
        nib.save(
            nib.Nifti1Image(image_data.astype(np.float32), affine), image_nifti_path
        )

        # Save NumPy arrays
        np.save(os.path.join(paths["npy"], "prediction.npy"), prediction)
        np.save(os.path.join(paths["npy"], "image.npy"), image_data)
        logger.info(f"Saved NumPy arrays to: {paths['npy']}")

        # Save metadata
        with open(paths["metadata"], "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {paths['metadata']}")

    def process_series(
        self,
        model: torch.nn.Module,
        dicom_files: List[str],
        output_paths: Dict[str, str],
        series_id: str,
        side: str,
        is_split: bool,
        device: str = "cuda",
    ) -> Dict:
        """Process a single DICOM series"""
        start_time = time.time()

        # Load DICOM series
        logger.info(f"Loading DICOM series: {series_id}")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        # Convert to numpy
        image_data = sitk.GetArrayFromImage(image)
        image_data = np.transpose(image_data, (2, 1, 0))  # Reorder axes

        # Get spacing and affine
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()

        # Create affine matrix
        affine = np.eye(4)
        affine[:3, :3] = np.array(direction).reshape(3, 3) * np.array(spacing)
        affine[:3, 3] = origin

        original_shape = image_data.shape
        total_slices = original_shape[2]

        # Handle splitting if requested
        if is_split:
            left_data, right_data = self.split_left_right(image_data)

            if side.lower() == "left":
                image_data = left_data
            elif side.lower() == "right":
                image_data = right_data
            elif side.lower() == "both":
                # Process both sides separately (this would need recursive calls)
                logger.info(
                    "Processing both sides separately not implemented in this function"
                )
                image_data = image_data  # Use full image for now

        # Run inference
        prediction, inference_time = self.run_inference_on_image(
            model, image_data, device
        )

        # Calculate metrics
        volume_metrics = self.calculate_volume_metrics(prediction, spacing)

        # Prepare metadata
        metadata = {
            "series_id": series_id,
            "side": side,
            "processing_timestamp": datetime.now().isoformat(),
            "total_processing_time": time.time() - start_time,
            "inference_time": inference_time,
            "original_shape": original_shape,
            "processed_shape": image_data.shape,
            "prediction_shape": prediction.shape,
            "total_slices": total_slices,
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
            "is_split": is_split,
            "volume_metrics": volume_metrics,
            "dicom_files_count": len(dicom_files),
            "unique_classes": [int(x) for x in np.unique(prediction)],
        }

        # Save results
        self.save_results(
            output_paths, image_data, prediction, affine, spacing, metadata
        )

        logger.info(
            f"Completed processing {series_id} ({side}) in {metadata['total_processing_time']:.2f} seconds"
        )

        return metadata


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Medical Image Inference Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth

  # With splitting and specific side
  python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth --split --side left

  # Custom model name and minimum slices
  python main2.py --dcm_dir /path/to/dicom --out_dir /path/to/output --model_path /path/to/model.pth --model_name CustomUNet --min_slices 15
        """,
    )

    # Required arguments
    parser.add_argument(
        "--dcm_dir", type=str, required=True, help="Path to DICOM directory"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model file (.pth)",
    )

    # Optional arguments
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="Split image into left and right parts (default: False)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="UNet3D",
        help="Model name for output directory structure (default: UNet3D)",
    )
    parser.add_argument(
        "--min_slices",
        type=int,
        default=10,
        help="Minimum number of slices required for processing (default: 10)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which side to process when splitting (default: both)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (cuda/cpu) (default: cuda)",
    )

    return parser


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize CLI
    cli = InferenceCLI()

    try:
        # Validate arguments
        if not torch.cuda.is_available() and args.device == "cuda":
            logger.warning("CUDA not available, switching to CPU")
            args.device = "cpu"

        # Load model
        model = cli.load_model(args.model_path, args.device)

        # Process DICOM directory
        series_list = cli.process_dicom_directory(args.dcm_dir, args.min_slices)

        if not series_list:
            logger.error("No valid DICOM series found")
            return 1

        # Process each series
        all_results = []

        for series_id, dicom_files in series_list:
            logger.info(f"Processing series: {series_id}")

            # Determine sides to process
            sides_to_process = (
                ["both"]
                if not args.split
                else ([args.side] if args.side != "both" else ["left", "right"])
            )

            for side in sides_to_process:
                try:
                    # Create output structure
                    output_paths = cli.create_output_structure(
                        args.out_dir, args.model_name, series_id, side
                    )

                    # Process series
                    metadata = cli.process_series(
                        model=model,
                        dicom_files=dicom_files,
                        output_paths=output_paths,
                        series_id=series_id,
                        side=side,
                        is_split=args.split,
                        device=args.device,
                    )

                    all_results.append(metadata)

                except Exception as e:
                    logger.error(f"Failed to process series {series_id} ({side}): {e}")
                    continue

        # Save summary
        summary_path = os.path.join(args.out_dir, "processing_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "arguments": vars(args),
                    "total_series_processed": len(all_results),
                    "processing_timestamp": datetime.now().isoformat(),
                    "results": all_results,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Processing completed. Summary saved to: {summary_path}")
        logger.info(f"Total series processed: {len(all_results)}")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
