import os
import pydicom
import numpy as np
import SimpleITK as sitk

def load_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image, dicom_names

def keep_only_femur(femur_and_other_img, other_bones_only_img):
    """
    Subtract 'other bones only' from 'femur + other bones' to isolate femur.
    Returns a new image where only the femur remains.
    """
    femur_mask = sitk.Subtract(femur_and_other_img, other_bones_only_img)
    femur_mask = sitk.Cast(femur_mask > 0, sitk.sitkUInt8)  # Binary mask
    return sitk.Mask(femur_and_other_img, femur_mask)

def save_dicom_series(image, reference_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    writer = sitk.ImageFileWriter()
    
    for i in range(image.GetDepth()):
        try:
            slice_i = image[:, :, i]
            # Convert to numpy array
            array = sitk.GetArrayFromImage(slice_i)
            
            # Handle different array shapes
            if array.ndim == 3:
                array = array[0]  # Remove singleton dimension
            elif array.ndim == 1:
                # Reconstruct 2D array from 1D
                array = array.reshape(image.GetHeight(), image.GetWidth())
            
            array = array.astype(np.uint16)
            
            # Read reference DICOM with force=True to handle non-compliant headers
            ref_ds = pydicom.dcmread(reference_files[i], force=True)
            
            # Update pixel data
            ref_ds.PixelData = array.tobytes()
            ref_ds.Rows, ref_ds.Columns = array.shape
            ref_ds.BitsAllocated = 16
            ref_ds.BitsStored = 16
            ref_ds.HighBit = 15
            ref_ds.SamplesPerPixel = 1
            ref_ds.PhotometricInterpretation = "MONOCHROME2"
            
            # Optional: Update other fields
            ref_ds.InstanceNumber = i + 1
            
            # Save to file
            output_path = os.path.join(output_dir, f"{i:04d}.dcm")
            ref_ds.save_as(output_path)
            
        except Exception as e:
            print(f"Warning: Error processing slice {i}: {e}")
            continue

def main():
    femur_and_other_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/final/CT3/RT023_left_CT3_prediction_class1_dcm"
    other_bones_only_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/final/CT3/RT023_left_CT3_prediction_class2_dcm"
    output_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/out/CT3/RT023_left_CT3_prediction_class1_dcm"
    
    print("Loading DICOM series (femur + other bones)...")
    femur_and_other_img, femur_ref = load_dicom_series(femur_and_other_dir)
    
    print("Loading DICOM series (other bones only)...")
    other_bones_only_img, _ = load_dicom_series(other_bones_only_dir)
    
    print("Isolating femur...")
    femur_only_img = keep_only_femur(femur_and_other_img, other_bones_only_img)
    
    print(f"Saving result to {output_dir}...")
    save_dicom_series(femur_only_img, femur_ref, output_dir)
    
    print("✅ Done.")

if __name__ == "__main__":
    main()