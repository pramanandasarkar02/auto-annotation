import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_and_save_side_by_side_text(image_slice, label_slice, title, output_dir="output"):
    """Save image slice data and optionally visualize side by side with label."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract slice number from title for specific slice saving
    if "Slice 200" in title:  # This is slice 120 (0-indexed)
        slice_filename = f"slice_120_values.txt"
        slice_path = os.path.join(output_dir, slice_filename)
        np.savetxt(slice_path, image_slice, fmt="%.4f")
        print(f"Saved slice 120 values to {slice_path}")
    return
    
    # Optional: Create and save visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show image slice
    im1 = axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Show label slice
    im2 = axes[1].imshow(label_slice, cmap='jet')
    axes[1].set_title('Label')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(title)
    
    # Save figure
    safe_title = title.replace(" | ", "_").replace(" ", "_")
    fig_path = os.path.join(output_dir, f"{safe_title}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    return fig_path

def load_images_labels(image_path, label_path):
    """Load NIfTI image and label files."""
    try:
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        print(f"Loaded {os.path.basename(image_path)} | Image shape: {image.shape}, Label shape: {label.shape}")
        
        # Verify shapes match
        if image.shape != label.shape:
            print(f"Warning: Shape mismatch - Image: {image.shape}, Label: {label.shape}")
            
        return image, label
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

def show_image_and_label(image, label, series_id=None, patient_id=None, output_dir="output"):
    """Process all slices of an image-label pair."""
    if image is None or label is None:
        print("Skipping processing due to loading errors")
        return
        
    max_slices = image.shape[2]
    print(f"Processing {max_slices} slices...")
    
    for i in range(max_slices):
        image_slice = image[:, :, i]
        label_slice = label[:, :, i]
        title = f"{series_id} | {patient_id} | Slice {i}"
        
        # Only save visualization for slices with actual content
        if np.any(image_slice) or np.any(label_slice):
            show_and_save_side_by_side_text(image_slice, label_slice, title, output_dir)

def visualize_image_label(images_dir, labels_dir, series_id, patient_id=None, output_dir="output"):
    """Main function to process medical images and labels."""
    image_series_path = os.path.join(images_dir, series_id)
    label_series_path = os.path.join(labels_dir, series_id)
    
    print(f"Image path: {image_series_path}")
    print(f"Label path: {label_series_path}")
    
    if not os.path.isdir(image_series_path):
        print(f"Image directory not found: {image_series_path}")
        return
    if not os.path.isdir(label_series_path):
        print(f"Label directory not found: {label_series_path}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(image_series_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    for image_file in image_files:
        if patient_id and patient_id not in image_file:
            continue
            
        image_path = os.path.join(image_series_path, image_file)
        
        # Find corresponding label file
        label_candidates = [f for f in os.listdir(label_series_path) 
                          if patient_id in f and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        
        if not label_candidates:
            print(f"No label found for {image_file} in {label_series_path}")
            continue
            
        label_path = os.path.join(label_series_path, label_candidates[0])
        print(f"Processing pair: {image_path} <-> {label_path}")
        
        # Load and process
        image, label = load_images_labels(image_path, label_path)
        if image is not None and label is not None:
            show_image_and_label(image, label, series_id, image_file, output_dir)

def analyze_slice_statistics(image, label, slice_idx=119):
    """Analyze statistics for a specific slice."""
    if slice_idx >= image.shape[2]:
        print(f"Slice {slice_idx} not available. Max slice: {image.shape[2]-1}")
        return
        
    image_slice = image[:, :, slice_idx]
    label_slice = label[:, :, slice_idx]
    
    print(f"\nSlice {slice_idx} Statistics:")
    print(f"Image - Min: {image_slice.min():.4f}, Max: {image_slice.max():.4f}, Mean: {image_slice.mean():.4f}")
    print(f"Label - Unique values: {np.unique(label_slice)}")
    print(f"Image slice shape: {image_slice.shape}")
    print(f"Non-zero pixels in image: {np.count_nonzero(image_slice)}")
    print(f"Non-zero pixels in label: {np.count_nonzero(label_slice)}")

if __name__ == "__main__":
    # Configuration
    root_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data"
    image_dir = os.path.join(root_dir, "image")
    label_dir = os.path.join(root_dir, "label")
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process specific series and patient
    series_id = "CT3"
    patient_id = "RG018_left"
    
    print(f"Starting processing for Series: {series_id}, Patient: {patient_id}")
    visualize_image_label(image_dir, label_dir, series_id, patient_id, output_dir)
    
    print("Processing complete!")