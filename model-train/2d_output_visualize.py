import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import os

def visualize_image_label_slice(image_slice, label_slice, slice_idx=0, output_file="output.txt", show_plot=True):
    """
    Visualize and analyze a single slice of medical image and its corresponding label.
    Prints detailed statistics to output file and displays visualization.
    
    Args:
        image_slice (numpy.ndarray): 2D image slice
        label_slice (numpy.ndarray): 2D label slice
        slice_idx (int): Index of the slice being analyzed
        output_file (str): File to write analysis results
        show_plot (bool): Whether to display the plot
    """
    
    # Calculate statistics
    image_stats = {
        'min': np.min(image_slice),
        'max': np.max(image_slice),
        'mean': np.mean(image_slice),
        'std': np.std(image_slice),
        'shape': image_slice.shape
    }
    
    # Ensure label slice is integer type for proper analysis
    label_slice_int = label_slice.astype(int)
    unique_labels = np.unique(label_slice_int)
    
    label_stats = {
        'unique_labels': unique_labels,
        'label_counts': np.bincount(label_slice_int.flatten(), minlength=len(unique_labels)),
        'shape': label_slice.shape
    }
    
    # Write to output file
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"SLICE {slice_idx} ANALYSIS\n")
        f.write(f"{'='*50}\n")
        f.write(f"Image shape: {image_stats['shape']}\n")
        f.write(f"Image min/max: {image_stats['min']:.4f} / {image_stats['max']:.4f}\n")
        f.write(f"Image mean ± std: {image_stats['mean']:.4f} ± {image_stats['std']:.4f}\n\n")
        
        f.write(f"LABEL SLICE ANALYSIS\n")
        f.write(f"Label shape: {label_stats['shape']}\n")
        f.write(f"Unique labels: {label_stats['unique_labels']}\n")
        f.write(f"Label distribution:\n")
        
        total_pixels = np.prod(label_slice.shape)
        for label_val in unique_labels:
            count = np.sum(label_slice_int == label_val)
            percentage = (count / total_pixels) * 100
            f.write(f"  Label {label_val}: {count} pixels ({percentage:.2f}%)\n")
        f.write(f"\n")
    
    # Create visualization if requested
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        im1 = axes[0].imshow(image_slice, cmap='gray', interpolation='nearest')
        axes[0].set_title(f'Original Image - Slice {slice_idx}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.6)
        
        # Label mask
        # Create custom colormap for labels
        n_labels = len(unique_labels)
        if n_labels <= 1:
            colors = [[0, 0, 0, 1]]  # Just black for background
        else:
            colors = plt.cm.Set1(np.linspace(0, 1, max(n_labels, 3)))[:n_labels]
            colors[0] = [0, 0, 0, 1]  # Background as black
        
        cmap_labels = ListedColormap(colors)
        
        im2 = axes[1].imshow(label_slice_int, cmap=cmap_labels, interpolation='nearest', 
                            vmin=0, vmax=max(unique_labels) if len(unique_labels) > 0 else 1)
        axes[1].set_title(f'Label Mask - Slice {slice_idx}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.6)
        
        # Overlay
        axes[2].imshow(image_slice, cmap='gray', alpha=0.7)
        overlay_mask = label_slice_int > 0  # Only show non-background labels
        if np.any(overlay_mask):
            axes[2].imshow(np.ma.masked_where(~overlay_mask, label_slice_int), 
                          cmap=cmap_labels, alpha=0.5, interpolation='nearest',
                          vmin=0, vmax=max(unique_labels) if len(unique_labels) > 0 else 1)
        axes[2].set_title(f'Overlay - Slice {slice_idx}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure with slice-specific name
        output_filename = f'slice_analysis_{slice_idx:03d}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved as {output_filename}")
        
        plt.show()

    print(f"Slice {slice_idx} analysis written to {output_file}")

def show_image_and_label(image, label, series_id=None, patient_id=None, 
                        max_slices_to_show=None, show_middle_only=False):
    """
    Display multiple slices of image and label data.
    
    Args:
        image: 3D image array
        label: 3D label array
        series_id: Series identifier for display
        patient_id: Patient identifier for display
        max_slices_to_show: Maximum number of slices to process (None for all)
        show_middle_only: If True, only show the middle slice
    """
    
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Image data type: {image.dtype}")
    print(f"Label data type: {label.dtype}")
    
    max_slices = image.shape[2]
    
    if show_middle_only:
        # Only show middle slice
        slice_idx = max_slices // 2
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]
        
        print(f"Analyzing middle slice {slice_idx} of {max_slices}")
        visualize_image_label_slice(image_slice, label_slice, slice_idx)
        
    else:
        # Show multiple slices
        if max_slices_to_show is not None:
            max_slices = min(max_slices, max_slices_to_show)
        
        print(f"Processing {max_slices} slices...")
        
        for i in range(max_slices):
            image_slice = image[:, :, i]
            label_slice = label[:, :, i]
            
            # Check if slice has any meaningful data
            if np.any(label_slice > 0) or max_slices % 30 == 0 :  # Show slices with labels or middle slice
                print(f"Processing slice {i}/{max_slices-1}")
                show_plot = (max_slices % 30 == 0)  # Only show plot for middle slice to avoid too many windows
                visualize_image_label_slice(image_slice, label_slice, i, show_plot=show_plot)


if __name__ == "__main__":
    # File paths
    # image_path = "/home/pramananda/working_dir/qct-medical-image-auto-annotation/dataset/femur-bone/data/image/CT3/RG018_right_CT3.nii.gz"
    # label_path = "/home/pramananda/working_dir/qct-medical-image-auto-annotation/dataset/femur-bone/data/label/CT3/RG018_right_CT3.nii.gz"

    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT2/RG018_left_CT2.nii.gz"
    # label_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/label/CT2/RT023_left_CT2.nii.gz"
    pred_path = "/home/user/auto-annotation/auto-annotation/predictions/RG018_right_CT2_prediction.nii.gz"
    label_path = pred_path
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        exit(1)
    
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        exit(1)
    
    try:
        # Load data
        print("Loading image and label data...")
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # If label is prediction with multiple channels, convert it
        if label.ndim == 4 and label.shape[0] <= 10:
            print(f"Detected multi-class prediction with shape {label.shape}")
            label = np.argmax(label, axis=0)
            print(f"Post-argmax label shape: {label.shape}")
        
        # Clear previous output file
        output_file = "output.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # Show analysis - you can modify these parameters as needed
        show_image_and_label(
            image, 
            label, 
            series_id="CT3",
            patient_id="RG018_right",
            max_slices_to_show=10,  # Limit to 10 slices to avoid too much output
            show_middle_only=True   # Set to True to only analyze middle slice
        )
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        import traceback
        traceback.print_exc()