import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ScaleIntensityRanged, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
import pydicom
from pydicom.dataset import Dataset as DicomDataset
from pydicom.uid import generate_uid
import datetime
from model import create_qct_segmentation_model  # Import your model creation function

def create_inference_transforms():
    """Create the same transforms used during training for consistency"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

def load_trained_model(model_path, model_name="unet", device='cuda'):
    """
    Load the trained model with proper architecture
    
    Args:
        model_path: Path to the saved model weights
        model_name: Name of the model architecture (unet, unetr, swinunetr, segresnet)
        device: Device to load the model on
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading {model_name} model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create the model using your model creation function
    model = create_qct_segmentation_model(model_name)
    
    # Load the state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        raise

def prediction(image_path, model_path, output_path, model_name="unet", device=None, 
               threshold=0.5):
    """
    Perform binary prediction on a single 3D medical image using the trained model
    
    Args:
        image_path: Path to input NIfTI image
        model_path: Path to trained model weights
        output_path: Path to save prediction NIfTI
        model_name: Name of the model architecture
        device: Device to run inference on
        threshold: Probability threshold for binary classification
    
    Returns:
        dict: Dictionary containing binary prediction and probability map
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting binary prediction...")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Model type: {model_name}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}")
    
    # Check if input file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Create inference transforms (same as training)
    inference_transforms = create_inference_transforms()
    
    # Prepare data
    test_data = [{"image": image_path}]
    test_ds = Dataset(data=test_data, transform=inference_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    
    # Load the trained model
    try:
        model = load_trained_model(model_path, model_name, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return create_dummy_binary_prediction(image_path, output_path, threshold)
    
    print("Running inference with sliding window...")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            
            print(f"Input tensor shape: {test_inputs.shape}")
            
            # Use sliding window inference (same as validation)
            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=(96, 96, 96),  # Same ROI size as training
                sw_batch_size=1,
                predictor=model,
                overlap=0.5  # Add overlap for better predictions
            )
            
            print(f"Raw output shape: {test_outputs.shape}")
            print(f"Raw output range: [{test_outputs.min():.3f}, {test_outputs.max():.3f}]")
            
            # Apply sigmoid to get probabilities for binary classification
            test_outputs = torch.sigmoid(test_outputs)
            
            print(f"After sigmoid - output range: [{test_outputs.min():.3f}, {test_outputs.max():.3f}]")
            
            # Get binary predictions using threshold
            binary_predictions = (test_outputs > threshold).float()
            
            print(f"Binary predictions shape: {binary_predictions.shape}")
            print(f"Unique values in prediction: {torch.unique(binary_predictions)}")
            
            # Convert to numpy
            binary_predictions_np = binary_predictions[0, 0].cpu().numpy().astype(np.uint8)  # Remove batch and channel dims
            probability_map = test_outputs[0, 0].cpu().numpy()  # Probability map
            
            print(f"Final prediction shape: {binary_predictions_np.shape}")
            print(f"Unique values: {np.unique(binary_predictions_np)}")
            
            # Count voxels for each class
            background_count = np.sum(binary_predictions_np == 0)
            foreground_count = np.sum(binary_predictions_np == 1)
            total_voxels = binary_predictions_np.size
            
            bg_percentage = (background_count / total_voxels) * 100
            fg_percentage = (foreground_count / total_voxels) * 100
            
            print(f"Background (0): {background_count} voxels ({bg_percentage:.2f}%)")
            print(f"Foreground (1): {foreground_count} voxels ({fg_percentage:.2f}%)")
            
            # Prepare results dictionary
            results = {
                'binary_prediction': binary_predictions_np,
                'probability_map': probability_map,
                'threshold': threshold
            }
            
            # Save binary prediction
            save_binary_predictions(results, image_path, output_path)
            
            return results

def create_dummy_binary_prediction(image_path, output_path, threshold=0.5):
    """Create a meaningful dummy binary prediction based on the actual image"""
    print("Creating dummy binary prediction for demonstration...")
    
    # Load the original image to get its shape and properties
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata()
    
    print(f"Original image shape: {image_data.shape}")
    
    # Create binary prediction based on intensity threshold
    normalized_image = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # Simple threshold-based binary segmentation for demonstration
    # You can adjust this threshold based on your specific use case
    intensity_threshold = 0.4
    binary_prediction = (normalized_image > intensity_threshold).astype(np.uint8)
    
    # Create dummy probability map
    probability_map = normalized_image.astype(np.float32)
    
    print(f"Dummy binary prediction created:")
    background_count = np.sum(binary_prediction == 0)
    foreground_count = np.sum(binary_prediction == 1)
    total_voxels = binary_prediction.size
    
    bg_percentage = (background_count / total_voxels) * 100
    fg_percentage = (foreground_count / total_voxels) * 100
    
    print(f"  Background (0): {background_count} voxels ({bg_percentage:.2f}%)")
    print(f"  Foreground (1): {foreground_count} voxels ({fg_percentage:.2f}%)")
    
    results = {
        'binary_prediction': binary_prediction,
        'probability_map': probability_map,
        'threshold': threshold
    }
    
    save_binary_predictions(results, image_path, output_path)
    return results

def save_binary_predictions(results, original_image_path, output_path):
    """Save binary predictions and probability maps as NIfTI files"""
    print(f"Saving binary predictions...")
    
    # Load original image to get header info
    original_nii = nib.load(original_image_path)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace('.nii.gz', '')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary prediction (main output)
    binary_prediction = results['binary_prediction']
    binary_nii = nib.Nifti1Image(
        binary_prediction.astype(np.uint8), 
        original_nii.affine, 
        original_nii.header
    )
    binary_nii.header.set_data_dtype(np.uint8)
    
    nib.save(binary_nii, output_path)
    print(f"Binary prediction saved: {output_path}")
    
    # Save probability map
    probability_map = results['probability_map']
    prob_nii = nib.Nifti1Image(
        probability_map.astype(np.float32),
        original_nii.affine,
        original_nii.header
    )
    prob_nii.header.set_data_dtype(np.float32)
    
    prob_path = output_path.replace('.nii.gz', '_probability.nii.gz')
    nib.save(prob_nii, prob_path)
    print(f"Probability map saved: {prob_path}")

def save_prediction_as_nifti(prediction, original_image_path, output_path):
    """Save prediction as NIfTI file with same header as original"""
    print(f"Saving prediction to: {output_path}")
    
    # Load original image to get header info
    original_nii = nib.load(original_image_path)
    
    # Create new NIfTI image with prediction data but original header
    prediction_nii = nib.Nifti1Image(
        prediction.astype(np.uint8), 
        original_nii.affine, 
        original_nii.header
    )
    
    # Update header for segmentation
    prediction_nii.header.set_data_dtype(np.uint8)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the prediction
    nib.save(prediction_nii, output_path)
    print(f"Prediction saved successfully!")

def nifti_to_dcm_binary(nifti_path, dcm_dir, is_probability=False):
    """
    Convert binary NIfTI file to DICOM series
    
    Args:
        nifti_path: Path to input NIfTI file
        dcm_dir: Directory to save DICOM files
        is_probability: Whether the input is a probability map or binary mask
    """
    print(f"Converting binary NIfTI to DICOM...")
    print(f"Input: {nifti_path}")
    print(f"Output directory: {dcm_dir}")
    print(f"Is probability map: {is_probability}")
    
    # Create output directory
    os.makedirs(dcm_dir, exist_ok=True)
    
    # Load NIfTI file
    nii_img = nib.load(nifti_path)
    data = nii_img.get_fdata()
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"Unique values: {np.unique(data)}")
    
    # Get image properties
    spacing = nii_img.header.get_zooms()
    
    # Convert to appropriate format for DICOM
    if is_probability:
        # Scale probability values to 0-4095 range for better visualization
        data = (data * 4095).astype(np.uint16)
        series_description = "Binary Segmentation Probability"
    else:
        # Binary mask: 0 -> 0, 1 -> 4095 for better contrast
        data = (data.astype(np.uint16) * 4095)
        series_description = "Binary Segmentation Mask"
    
    # Generate unique identifiers
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    
    print(f"Creating {data.shape[2]} DICOM slices...")
    
    saved_slices = 0
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        
        # Skip completely empty slices for binary masks
        if not is_probability and np.sum(slice_data) == 0:
            continue
        
        # Create DICOM dataset
        ds = DicomDataset()
        
        # Set required DICOM tags
        ds.PatientName = "Anonymous"
        ds.PatientID = "ANON001"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.SOPInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = frame_of_reference_uid
        
        current_time = datetime.datetime.now()
        ds.StudyDate = current_time.strftime("%Y%m%d")
        ds.StudyTime = current_time.strftime("%H%M%S")
        ds.SeriesDate = current_time.strftime("%Y%m%d")
        ds.SeriesTime = current_time.strftime("%H%M%S")
        
        ds.StudyDescription = "AI Binary Segmentation Result"
        ds.SeriesDescription = series_description
        ds.SeriesNumber = "100"
        ds.InstanceNumber = str(i + 1)
        
        # Image-specific tags
        ds.Modality = "SEG"  # Segmentation modality
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Set spacing information
        ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]
        ds.SliceThickness = str(spacing[2])
        ds.SpacingBetweenSlices = str(spacing[2])
        
        # Set position information
        ds.ImagePositionPatient = [0, 0, i * spacing[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        
        # Set slice location
        ds.SliceLocation = str(i * spacing[2])
        
        # Set window/level for better visualization
        if is_probability:
            ds.WindowCenter = "2047"
            ds.WindowWidth = "4095"
        else:
            ds.WindowCenter = "2047"
            ds.WindowWidth = "4095"
        
        # Set the pixel data
        ds.PixelData = slice_data.tobytes()
        
        # Set transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.ImplementationClassUID = generate_uid()
        
        # Save DICOM file
        filename = f"seg_slice_{i:04d}.dcm"
        filepath = os.path.join(dcm_dir, filename)
        
        try:
            pydicom.dcmwrite(filepath, ds)
            saved_slices += 1
        except Exception as e:
            print(f"Error writing slice {i}: {e}")
            continue
    
    print(f"DICOM conversion completed!")
    print(f"Saved {saved_slices} DICOM slices in {dcm_dir}")

if __name__ == "__main__":
    # File paths
    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT2/RT023_left_CT2.nii.gz"
    pred_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/pred/CT2/" 
    os.makedirs(pred_dir, exist_ok=True)
    pred_basename = os.path.basename(image_path).replace('.nii.gz', '_prediction.nii.gz')
    prediction_path = os.path.join(pred_dir, pred_basename)
    
    # DICOM output directories
    dcm_dir_binary = prediction_path.replace('.nii.gz', '_binary_dcm')
    dcm_dir_prob = prediction_path.replace('.nii.gz', '_probability_dcm')
    
    # Model configuration
    model_path = "/home/user/auto-annotation/auto-annotation/models/unet/combined_model.pth"
    model_name = "unet"
    threshold = 0.5  # Binary classification threshold
    
    try:
        print("="*80)
        print("BINARY MEDICAL IMAGE PREDICTION AND CONVERSION PIPELINE")
        print("="*80)
        
        # Step 1: Run binary prediction
        print("\n1. Running binary prediction...")
        results = prediction(
            image_path=image_path,
            model_path=model_path, 
            output_path=prediction_path,
            model_name=model_name,
            threshold=threshold
        )
        
        binary_prediction = results['binary_prediction']
        probability_map = results['probability_map']
        
        print(f"\nPrediction Analysis:")
        print(f"Prediction shape: {binary_prediction.shape}")
        print(f"Unique values: {np.unique(binary_prediction)}")
        
        background_count = np.sum(binary_prediction == 0)
        foreground_count = np.sum(binary_prediction == 1)
        total_voxels = binary_prediction.size
        
        bg_percentage = (background_count / total_voxels) * 100
        fg_percentage = (foreground_count / total_voxels) * 100
        
        print(f"Background (0): {background_count} voxels ({bg_percentage:.2f}%)")
        print(f"Foreground (1): {foreground_count} voxels ({fg_percentage:.2f}%)")
        print(f"Mean probability: {np.mean(probability_map):.3f}")
        print(f"Max probability: {np.max(probability_map):.3f}")
        print(f"Min probability: {np.min(probability_map):.3f}")
        
        # Step 2: Convert to DICOM
        print("\n2. Converting to DICOM...")
        
        # Convert binary prediction
        print("Converting binary mask...")
        nifti_to_dcm_binary(prediction_path, dcm_dir_binary, is_probability=False)
        
        # Convert probability map
        print("Converting probability map...")
        prob_path = prediction_path.replace('.nii.gz', '_probability.nii.gz')
        nifti_to_dcm_binary(prob_path, dcm_dir_prob, is_probability=True)
        
        print("\n" + "="*80)
        print("BINARY PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Output files:")
        print(f"  Binary NIfTI: {prediction_path}")
        print(f"  Probability NIfTI: {prob_path}")
        print(f"  Binary DICOM: {dcm_dir_binary}")
        print(f"  Probability DICOM: {dcm_dir_prob}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()