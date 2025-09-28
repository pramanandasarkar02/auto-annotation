import pydicom
import numpy as np
import os
from pathlib import Path
import SimpleITK as sitk
from scipy import ndimage

def load_dicom_series(dicom_dir):
    """Load DICOM series from directory"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def create_bone_mask(image, threshold_min=200, threshold_max=3000):
    """Create binary mask for bone tissue based on HU values"""
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Create bone mask (typical bone HU values: 200-3000)
    bone_mask = np.logical_and(array >= threshold_min, array <= threshold_max)
    
    # Convert back to SimpleITK image
    mask_image = sitk.GetImageFromArray(bone_mask.astype(np.uint8))
    mask_image.CopyInformation(image)
    
    return mask_image

def subtract_masks(femur_and_other_image, other_bones_only_image, threshold=200):
    """
    Subtract other bones from femur+other bones to isolate femur
    
    Args:
        femur_and_other_image: DICOM with femur + other bones
        other_bones_only_image: DICOM with only other bones (no femur)
        threshold: HU threshold for bone detection
    """
    
    # Convert to numpy arrays
    combined_array = sitk.GetArrayFromImage(femur_and_other_image)
    other_bones_array = sitk.GetArrayFromImage(other_bones_only_image)
    
    # Create bone masks
    combined_bone_mask = (combined_array >= threshold).astype(np.uint8)
    other_bones_mask = (other_bones_array >= threshold).astype(np.uint8)
    
    # Subtract other bones mask from combined mask
    # This should leave only femur
    femur_only_mask = combined_bone_mask - other_bones_mask
    femur_only_mask = np.clip(femur_only_mask, 0, 1)  # Ensure binary values
    
    # Apply morphological operations to clean up
    # Remove small artifacts
    femur_only_mask = ndimage.binary_opening(femur_only_mask, iterations=2)
    femur_only_mask = ndimage.binary_closing(femur_only_mask, iterations=2)
    
    # Apply mask to original image
    femur_only_array = combined_array * femur_only_mask
    
    # Convert back to SimpleITK image
    femur_only_image = sitk.GetImageFromArray(femur_only_array)
    femur_only_image.CopyInformation(femur_and_other_image)
    
    return femur_only_image, femur_only_mask

def save_dicom_series(image, output_dir, original_dir):
    """Save processed image as DICOM series"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original DICOM files for metadata
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(original_dir)
    
    # Get the array from the image
    array = sitk.GetArrayFromImage(image)
    
    # Write each slice as a separate DICOM file
    for i in range(array.shape[0]):
        # Extract single slice
        slice_array = array[i]
        slice_image = sitk.GetImageFromArray(slice_array)
        
        # Copy spacing and origin information from original image
        original_spacing = image.GetSpacing()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()
        
        # Set 2D spacing and origin for the slice
        slice_image.SetSpacing((original_spacing[0], original_spacing[1]))
        slice_image.SetOrigin((original_origin[0], original_origin[1]))
        # Set 2D direction matrix
        slice_image.SetDirection((original_direction[0], original_direction[1], 
                                original_direction[3], original_direction[4]))
        
        # Define output filename
        output_filename = os.path.join(output_dir, f"femur_only_{i:04d}.dcm")
        
        # Write the slice
        sitk.WriteImage(slice_image, output_filename)

def main():
    
    femur_and_other_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/final/CT3/RT023_left_CT3_prediction_class1_dcm"
    other_bones_only_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/final/CT3/RT023_left_CT3_prediction_class2_dcm"
    output_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/out/CT3/RT023_left_CT3_prediction_class1_dcm"

    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading DICOM series...")
    
    # Load both DICOM series
    try:
        femur_and_other = load_dicom_series(femur_and_other_dir)
        other_bones_only = load_dicom_series(other_bones_only_dir)
        print("DICOM series loaded successfully")
    except Exception as e:
        print(f"Error loading DICOM series: {e}")
        return
    
    # Check if dimensions match
    size1 = femur_and_other.GetSize()
    size2 = other_bones_only.GetSize()
    
    if size1 != size2:
        print(f"Warning: Image dimensions don't match!")
        print(f"Femur+Other: {size1}")
        print(f"Other bones only: {size2}")
        
        # Resample to match dimensions if needed
        print("Resampling to match dimensions...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(femur_and_other)
        other_bones_only = resampler.Execute(other_bones_only)
    
    print("Performing bone masking...")
    
    # Extract femur by subtracting other bones
    femur_only, femur_mask = subtract_masks(femur_and_other, other_bones_only)
    
    print("Saving results...")
    
    # Save the isolated femur as DICOM series
    save_dicom_series(femur_only, output_dir, femur_and_other_dir)
    
    # Optionally save the mask as well
    mask_output_dir = output_dir + "_mask"
    mask_image = sitk.GetImageFromArray(femur_mask.astype(np.int16))
    mask_image.CopyInformation(femur_and_other)
    save_dicom_series(mask_image, mask_output_dir, femur_and_other_dir)
    
    print(f"Femur extraction complete!")
    print(f"Output saved to: {output_dir}")
    print(f"Mask saved to: {mask_output_dir}")

# Alternative approach using registration if images are not aligned
def register_images(fixed_image, moving_image):
    """Register moving image to fixed image if they're not aligned"""
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100
    )
    registration_method.SetInitialTransform(initial_transform)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    
    registered_image = resampler.Execute(moving_image)
    return registered_image

if __name__ == "__main__":
    main()