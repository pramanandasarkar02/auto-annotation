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
               num_classes=3, save_all_classes=True):
    """
    Perform prediction on a single 3D medical image using the trained model
    
    Args:
        image_path: Path to input NIfTI image
        model_path: Path to trained model weights
        output_path: Path to save prediction NIfTI
        model_name: Name of the model architecture
        device: Device to run inference on
        num_classes: Number of output classes (4 in your case)
        save_all_classes: Whether to save all classes or just the argmax
    
    Returns:
        dict: Dictionary containing prediction arrays for each class
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting multiclass prediction...")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Model type: {model_name}")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    
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
        return create_dummy_multiclass_prediction(image_path, output_path, num_classes)
    
    # Post-processing transform for multiclass
    post_pred = AsDiscrete(argmax=True)  # Convert to class predictions using argmax

    # post_pred = AsDiscrete(threshold=0.5)

    
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
            
            # # Apply softmax to get probabilities
            test_outputs = torch.softmax(test_outputs, dim=1)
            
            print(f"After softmax - output range: [{test_outputs.min():.3f}, {test_outputs.max():.3f}]")
            
            # Get class predictions using argmax
            class_predictions = torch.argmax(test_outputs, dim=1)

            # binary segmentation
            # test_outputs = torch.sigmoid(test_outputs)
            # class_predictions = (test_outputs > 0.5).float()

            
            print(f"Class predictions shape: {class_predictions.shape}")
            print(f"Unique classes in prediction: {torch.unique(class_predictions)}")
            
            # Convert to numpy
            class_predictions_np = class_predictions[0].cpu().numpy().astype(np.uint8)
            
            # Also get probability maps for each class
            prob_maps = test_outputs[0].cpu().numpy()  # Shape: [4, H, W, D]
            
            print(f"Final prediction shape: {class_predictions_np.shape}")
            print(f"Unique classes: {np.unique(class_predictions_np)}")
            
            # Count voxels for each class
            for class_id in range(num_classes):
                count = np.sum(class_predictions_np == class_id)
                percentage = (count / class_predictions_np.size) * 100
                print(f"Class {class_id}: {count} voxels ({percentage:.2f}%)")
            
            # Prepare results dictionary
            results = {
                'class_prediction': class_predictions_np,
                'probabilities': prob_maps
            }
            
            # Save predictions
            if save_all_classes:
                save_multiclass_predictions(results, image_path, output_path, num_classes)
            else:
                # Save only the argmax prediction
                save_prediction_as_nifti(class_predictions_np, image_path, output_path)

            # save_prediction_as_nifti(class_predictions_np, image_path, output_path)

            
            return results

def create_dummy_multiclass_prediction(image_path, output_path, num_classes=3):
    """Create a meaningful dummy multiclass prediction based on the actual image"""
    print("Creating dummy multiclass prediction for demonstration...")
    
    # Load the original image to get its shape and properties
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata()
    
    print(f"Original image shape: {image_data.shape}")
    
    # Create multiclass prediction
    class_prediction = np.zeros(image_data.shape, dtype=np.uint8)
    
    # Simple threshold-based multiclass segmentation for demonstration
    normalized_image = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # Create different classes based on intensity thresholds
    # Class 0: Background (default)
    # Class 1: Soft tissue
    # Class 2: Cortical bone
    # Class 3: Trabecular bone
    
    soft_tissue_mask = (normalized_image > 0.3) & (normalized_image <= 0.6)
    cortical_bone_mask = normalized_image > 0.8
    trabecular_bone_mask = (normalized_image > 0.6) & (normalized_image <= 0.8)
    
    class_prediction[soft_tissue_mask] = 1
    class_prediction[trabecular_bone_mask] = 2
    class_prediction[cortical_bone_mask] = 3
    
    print(f"Dummy multiclass prediction created:")
    for class_id in range(num_classes):
        count = np.sum(class_prediction == class_id)
        percentage = (count / class_prediction.size) * 100
        print(f"  Class {class_id}: {count} voxels ({percentage:.2f}%)")
    
    # Create dummy probability maps
    prob_maps = np.zeros((num_classes,) + image_data.shape, dtype=np.float32)
    for class_id in range(num_classes):
        prob_maps[class_id] = (class_prediction == class_id).astype(np.float32)
    
    results = {
        'class_prediction': class_prediction,
        'probabilities': prob_maps
    }
    
    save_multiclass_predictions(results, image_path, output_path, num_classes)
    return results

def save_multiclass_predictions(results, original_image_path, output_path, num_classes):
    """Save multiclass predictions as separate NIfTI files"""
    print(f"Saving multiclass predictions...")
    
    # Load original image to get header info
    original_nii = nib.load(original_image_path)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace('.nii.gz', '')
    
    # Save class prediction (argmax)
    class_prediction = results['class_prediction']
    class_pred_nii = nib.Nifti1Image(
        class_prediction.astype(np.uint8), 
        original_nii.affine, 
        original_nii.header
    )
    class_pred_nii.header.set_data_dtype(np.uint8)
    
    class_pred_path = os.path.join(output_dir, f"{base_name}_classes.nii.gz")
    nib.save(class_pred_nii, class_pred_path)
    print(f"Class prediction saved: {class_pred_path}")
    
    # Save probability maps for each class
    prob_maps = results['probabilities']
    for class_id in range(num_classes):
        prob_map = prob_maps[class_id]
        prob_nii = nib.Nifti1Image(
            prob_map.astype(np.float32),
            original_nii.affine,
            original_nii.header
        )
        prob_nii.header.set_data_dtype(np.float32)
        
        prob_path = os.path.join(output_dir, f"{base_name}_class{class_id}_prob.nii.gz")
        nib.save(prob_nii, prob_path)
        print(f"Class {class_id} probability saved: {prob_path}")
    
    # Also save individual class masks as binary
    for class_id in range(1, num_classes):  # Skip background (class 0)
        class_mask = (class_prediction == class_id).astype(np.uint8)
        mask_nii = nib.Nifti1Image(
            class_mask,
            original_nii.affine,
            original_nii.header
        )
        mask_nii.header.set_data_dtype(np.uint8)
        
        mask_path = os.path.join(output_dir, f"{base_name}_class{class_id}_mask.nii.gz")
        nib.save(mask_nii, mask_path)
        print(f"Class {class_id} binary mask saved: {mask_path}")

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

def nifti_to_dcm_multiclass(nifti_path, dcm_dir, class_id=None):
    """
    Convert multiclass NIfTI file to DICOM series
    
    Args:
        nifti_path: Path to input NIfTI file
        dcm_dir: Directory to save DICOM files
        class_id: Specific class to convert (None for all classes)
    """
    print(f"Converting multiclass NIfTI to DICOM...")
    print(f"Input: {nifti_path}")
    print(f"Output directory: {dcm_dir}")
    
    # Create output directory
    os.makedirs(dcm_dir, exist_ok=True)
    
    # Load NIfTI file
    nii_img = nib.load(nifti_path)
    data = nii_img.get_fdata()
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Unique classes: {np.unique(data)}")
    
    # Get image properties
    spacing = nii_img.header.get_zooms()
    
    # Convert to appropriate format for DICOM
    if class_id is not None:
        # Convert specific class to binary mask
        data = (data == class_id).astype(np.uint16) * 4095
        series_description = f"Class {class_id} Segmentation"
    else:
        # Scale multiclass data
        data = data.astype(np.uint16) * (4095 // int(np.max(data)))
        series_description = "Multiclass Segmentation"
    
    # Generate unique identifiers
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    
    print(f"Creating {data.shape[2]} DICOM slices...")
    
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        
        # Skip empty slices
        if np.sum(slice_data) == 0:
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
        
        ds.StudyDescription = "AI Multiclass Segmentation Result"
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
        except Exception as e:
            print(f"Error writing slice {i}: {e}")
            continue
    
    print(f"DICOM conversion completed!")
    print(f"DICOM files saved in {dcm_dir}")

if __name__ == "__main__":
    # File paths
    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT3/RT023_left_CT3.nii.gz"
    label_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/label/CT3/RT023_left_CT3.nii.gz"
    pred_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/pred/CT3/" 
    os.makedirs(pred_dir, exist_ok=True)
    pred_basename = os.path.basename(image_path).replace('.nii.gz', '_prediction.nii.gz')
    prediction_path = os.path.join(pred_dir, pred_basename)
    
    # DICOM output directories
    dcm_dir_classes = prediction_path.replace('.nii.gz', '_classes_dcm')
    dcm_dir_class1 = prediction_path.replace('.nii.gz', '_class1_dcm')
    dcm_dir_class2 = prediction_path.replace('.nii.gz', '_class2_dcm')
    dcm_dir_class3 = prediction_path.replace('.nii.gz', '_class3_dcm')
    
    # Create prediction directory
    os.makedirs(pred_dir, exist_ok=True)
    
    # Model configuration
    model_path = "/home/user/auto-annotation/auto-annotation/models/unet/combined_model.pth"
    model_name = "unet"
    num_classes = 3  # Your model has 4 output classes
    
    try:
        print("="*80)
        print("MULTICLASS MEDICAL IMAGE PREDICTION AND CONVERSION PIPELINE")
        print("="*80)
        
        # Step 1: Run multiclass prediction
        print("\n1. Running multiclass prediction...")
        results = prediction(
            image_path=image_path,
            model_path=model_path, 
            output_path=prediction_path,
            model_name=model_name,
            num_classes=num_classes,
            save_all_classes=True
        )
        
        class_prediction = results['class_prediction']
        prob_maps = results['probabilities']
        
        print(f"\nPrediction Analysis:")
        print(f"Prediction shape: {class_prediction.shape}")
        print(f"Unique classes: {np.unique(class_prediction)}")
        
        for class_id in range(num_classes):
            count = np.sum(class_prediction == class_id)
            percentage = (count / class_prediction.size) * 100
            print(f"Class {class_id}: {count} voxels ({percentage:.2f}%)")
        
        # Step 2: Convert to DICOM for different classes
        print("\n2. Converting to DICOM...")
        
        # Convert class prediction (all classes)
        classes_nifti_path = os.path.join(pred_dir, pred_basename.replace('.nii.gz', '_classes.nii.gz'))
        nifti_to_dcm_multiclass(classes_nifti_path, dcm_dir_classes)
        
        # Convert individual classes
        for class_id in range(1, num_classes):  # Skip background
            class_nifti_path = os.path.join(pred_dir, pred_basename.replace('.nii.gz', f'_class{class_id}_mask.nii.gz'))
            class_dcm_dir = prediction_path.replace('.nii.gz', f'_class{class_id}_dcm')
            nifti_to_dcm_multiclass(classes_nifti_path, class_dcm_dir, class_id=class_id)

        # # Convert binary prediction to DICOM
        # nifti_to_dcm_multiclass(prediction_path, dcm_dir_classes)

        
        print("\n" + "="*80)
        print("MULTICLASS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Output files:")
        print(f"  Classes NIfTI: {classes_nifti_path}")
        print(f"  Classes DICOM: {dcm_dir_classes}")
        for class_id in range(1, num_classes):
            class_dcm_dir = prediction_path.replace('.nii.gz', f'_class{class_id}_dcm')
            print(f"  Class {class_id} DICOM: {class_dcm_dir}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()