import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import os

def analyze_slice_statistics(image_slice, label_slice, slice_idx):
    """
    Calculate detailed statistics for a single slice.
    
    Args:
        image_slice: 2D image array
        label_slice: 2D label array
        slice_idx: slice index
    
    Returns:
        dict: Dictionary containing all statistics
    """
    # Image statistics
    image_stats = {
        'min': np.min(image_slice),
        'max': np.max(image_slice),
        'mean': np.mean(image_slice),
        'std': np.std(image_slice),
        'median': np.median(image_slice),
        'shape': image_slice.shape,
        'non_zero_count': np.count_nonzero(image_slice),
        'zero_count': np.sum(image_slice == 0)
    }
    
    # Label statistics
    label_slice_int = label_slice.astype(int)
    unique_labels = np.unique(label_slice_int)
    total_pixels = np.prod(label_slice.shape)
    
    label_distribution = {}
    for label_val in unique_labels:
        count = np.sum(label_slice_int == label_val)
        percentage = (count / total_pixels) * 100
        label_distribution[label_val] = {
            'count': count,
            'percentage': percentage
        }
    
    label_stats = {
        'unique_labels': unique_labels,
        'num_unique_labels': len(unique_labels),
        'shape': label_slice.shape,
        'total_pixels': total_pixels,
        'distribution': label_distribution,
        'has_annotations': np.any(label_slice_int > 0),
        'annotation_pixels': np.sum(label_slice_int > 0),
        'annotation_percentage': (np.sum(label_slice_int > 0) / total_pixels) * 100
    }
    
    return {
        'slice_idx': slice_idx,
        'image': image_stats,
        'label': label_stats
    }

def comprehensive_image_analysis(image, label, output_file="comprehensive_analysis.txt"):
    """
    Perform comprehensive analysis of all slices in the 3D image and label data.
    
    Args:
        image: 3D image array
        label: 3D label array
        output_file: Output file for detailed analysis
    
    Returns:
        dict: Summary statistics for the entire volume
    """
    print(f"Starting comprehensive analysis...")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    
    num_slices = image.shape[2]
    all_slice_stats = []
    
    # Collect statistics for all slices
    for i in range(num_slices):
        image_slice = image[:, :, i]
        label_slice = label[:, :, i]
        
        slice_stats = analyze_slice_statistics(image_slice, label_slice, i)
        all_slice_stats.append(slice_stats)
        
        if (i + 1) % 10 == 0 or i == num_slices - 1:
            print(f"Processed {i + 1}/{num_slices} slices...")
    
    # Calculate volume-wide statistics
    volume_stats = calculate_volume_statistics(image, label, all_slice_stats)
    
    # Write detailed analysis to file
    write_comprehensive_analysis(all_slice_stats, volume_stats, output_file)
    
    # Display summary
    display_analysis_summary(volume_stats)
    
    return {
        'slice_statistics': all_slice_stats,
        'volume_statistics': volume_stats
    }

def calculate_volume_statistics(image, label, slice_stats):
    """Calculate statistics for the entire 3D volume."""
    
    # Overall image statistics
    image_volume_stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': np.min(image),
        'max': np.max(image),
        'mean': np.mean(image),
        'std': np.std(image),
        'median': np.median(image),
        'total_voxels': np.prod(image.shape),
        'non_zero_voxels': np.count_nonzero(image),
        'zero_voxels': np.sum(image == 0)
    }
    
    # Overall label statistics
    label_int = label.astype(int)
    unique_labels_volume = np.unique(label_int)
    total_voxels = np.prod(label.shape)
    
    label_volume_distribution = {}
    for label_val in unique_labels_volume:
        count = np.sum(label_int == label_val)
        percentage = (count / total_voxels) * 100
        label_volume_distribution[label_val] = {
            'count': count,
            'percentage': percentage
        }
    
    label_volume_stats = {
        'shape': label.shape,
        'dtype': str(label.dtype),
        'unique_labels': unique_labels_volume,
        'num_unique_labels': len(unique_labels_volume),
        'total_voxels': total_voxels,
        'distribution': label_volume_distribution,
        'annotated_voxels': np.sum(label_int > 0),
        'annotation_percentage': (np.sum(label_int > 0) / total_voxels) * 100
    }
    
    # Per-slice summary statistics
    slices_with_annotations = sum(1 for s in slice_stats if s['label']['has_annotations'])
    annotation_percentages = [s['label']['annotation_percentage'] for s in slice_stats if s['label']['has_annotations']]
    
    slice_summary = {
        'total_slices': len(slice_stats),
        'slices_with_annotations': slices_with_annotations,
        'slices_without_annotations': len(slice_stats) - slices_with_annotations,
        'annotation_coverage_percentage': (slices_with_annotations / len(slice_stats)) * 100,
        'avg_annotation_percentage_per_slice': np.mean(annotation_percentages) if annotation_percentages else 0,
        'max_annotation_percentage_per_slice': np.max(annotation_percentages) if annotation_percentages else 0,
        'min_annotation_percentage_per_slice': np.min(annotation_percentages) if annotation_percentages else 0
    }
    
    return {
        'image': image_volume_stats,
        'label': label_volume_stats,
        'slice_summary': slice_summary
    }

def write_comprehensive_analysis(slice_stats, volume_stats, output_file):
    """Write detailed analysis to output file."""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE 3D MEDICAL IMAGE AND LABEL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Volume-wide statistics
        f.write("VOLUME-WIDE STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        # Image volume stats
        img_stats = volume_stats['image']
        f.write(f"Image Volume:\n")
        f.write(f"  Shape: {img_stats['shape']}\n")
        f.write(f"  Data type: {img_stats['dtype']}\n")
        f.write(f"  Value range: {img_stats['min']:.4f} to {img_stats['max']:.4f}\n")
        f.write(f"  Mean ± Std: {img_stats['mean']:.4f} ± {img_stats['std']:.4f}\n")
        f.write(f"  Median: {img_stats['median']:.4f}\n")
        f.write(f"  Total voxels: {img_stats['total_voxels']:,}\n")
        f.write(f"  Non-zero voxels: {img_stats['non_zero_voxels']:,} ({(img_stats['non_zero_voxels']/img_stats['total_voxels']*100):.2f}%)\n\n")
        
        # Label volume stats
        label_stats = volume_stats['label']
        f.write(f"Label Volume:\n")
        f.write(f"  Shape: {label_stats['shape']}\n")
        f.write(f"  Data type: {label_stats['dtype']}\n")
        f.write(f"  Unique labels: {label_stats['unique_labels']}\n")
        f.write(f"  Total voxels: {label_stats['total_voxels']:,}\n")
        f.write(f"  Annotated voxels: {label_stats['annotated_voxels']:,} ({label_stats['annotation_percentage']:.2f}%)\n")
        f.write(f"  Label distribution across entire volume:\n")
        for label_val, info in label_stats['distribution'].items():
            f.write(f"    Label {label_val}: {info['count']:,} voxels ({info['percentage']:.2f}%)\n")
        f.write("\n")
        
        # Slice summary
        slice_summary = volume_stats['slice_summary']
        f.write(f"Slice Summary:\n")
        f.write(f"  Total slices: {slice_summary['total_slices']}\n")
        f.write(f"  Slices with annotations: {slice_summary['slices_with_annotations']}\n")
        f.write(f"  Slices without annotations: {slice_summary['slices_without_annotations']}\n")
        f.write(f"  Annotation coverage: {slice_summary['annotation_coverage_percentage']:.2f}% of slices\n")
        f.write(f"  Average annotation per annotated slice: {slice_summary['avg_annotation_percentage_per_slice']:.2f}%\n")
        f.write(f"  Max annotation per slice: {slice_summary['max_annotation_percentage_per_slice']:.2f}%\n")
        f.write(f"  Min annotation per slice: {slice_summary['min_annotation_percentage_per_slice']:.2f}%\n\n")
        
        # Detailed per-slice analysis
        f.write("="*80 + "\n")
        f.write("DETAILED PER-SLICE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for stats in slice_stats:
            slice_idx = stats['slice_idx']
            img = stats['image']
            lbl = stats['label']
            
            f.write(f"SLICE {slice_idx:03d}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Image - Shape: {img['shape']}, Range: [{img['min']:.4f}, {img['max']:.4f}], "
                   f"Mean±Std: {img['mean']:.4f}±{img['std']:.4f}\n")
            f.write(f"Label - Has annotations: {lbl['has_annotations']}, "
                   f"Annotated pixels: {lbl['annotation_pixels']}/{lbl['total_pixels']} "
                   f"({lbl['annotation_percentage']:.2f}%)\n")
            
            if lbl['has_annotations']:
                f.write(f"Label distribution:\n")
                for label_val, info in lbl['distribution'].items():
                    f.write(f"  Label {label_val}: {info['count']} pixels ({info['percentage']:.2f}%)\n")
            
            f.write("\n")

def display_analysis_summary(volume_stats):
    """Display a concise summary of the analysis."""
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    img_stats = volume_stats['image']
    label_stats = volume_stats['label']
    slice_summary = volume_stats['slice_summary']
    
    print(f"Image Volume: {img_stats['shape']} ({img_stats['dtype']})")
    print(f"Value Range: [{img_stats['min']:.4f}, {img_stats['max']:.4f}]")
    print(f"Mean ± Std: {img_stats['mean']:.4f} ± {img_stats['std']:.4f}")
    print()
    
    print(f"Label Volume: {label_stats['shape']} ({label_stats['dtype']})")
    print(f"Unique Labels: {list(label_stats['unique_labels'])}")
    print(f"Overall Annotation Coverage: {label_stats['annotation_percentage']:.2f}% of voxels")
    print()
    
    print(f"Slice Coverage: {slice_summary['slices_with_annotations']}/{slice_summary['total_slices']} slices have annotations ({slice_summary['annotation_coverage_percentage']:.2f}%)")
    print(f"Average annotation per annotated slice: {slice_summary['avg_annotation_percentage_per_slice']:.2f}%")
    print()
    
    print("Label Distribution (entire volume):")
    for label_val, info in label_stats['distribution'].items():
        label_name = "Background" if label_val == 0 else f"Annotation {label_val}"
        print(f"  {label_name}: {info['count']:,} voxels ({info['percentage']:.2f}%)")


def show_sample_slices(image, label, num_samples=3):
    """Show a few sample slices with annotations for visual verification."""
    
    # Find slices with annotations
    annotated_slices = []
    for i in range(image.shape[2]):
        if np.any(label[:, :, i] > 0):
            annotation_percentage = (np.sum(label[:, :, i] > 0) / np.prod(label[:, :, i].shape)) * 100
            annotated_slices.append((i, annotation_percentage))
    
    if not annotated_slices:
        print("No slices with annotations found!")
        return
    
    # Sort by annotation percentage and select diverse samples
    annotated_slices.sort(key=lambda x: x[1], reverse=True)
    
    # Select samples: highest, middle, and a random one
    sample_indices = []
    if len(annotated_slices) >= 3:
        sample_indices = [
            annotated_slices[0][0],  # Highest annotation
            annotated_slices[len(annotated_slices)//2][0],  # Middle
            annotated_slices[-1][0]  # Lowest annotation
        ]
    else:
        sample_indices = [slice_info[0] for slice_info in annotated_slices]
    
    sample_indices = sample_indices[:num_samples]
    
    print(f"\nShowing {len(sample_indices)} sample slices with annotations:")
    
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, slice_idx in enumerate(sample_indices):
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx].astype(int)
        
        annotation_pct = (np.sum(label_slice > 0) / np.prod(label_slice.shape)) * 100
        print(f"  Slice {slice_idx}: {annotation_pct:.2f}% annotated")
        
        # Original image
        axes[idx, 0].imshow(image_slice, cmap='gray')
        axes[idx, 0].set_title(f'Image - Slice {slice_idx}')
        axes[idx, 0].axis('off')
        
        # Label
        unique_labels = np.unique(label_slice)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        colors[0] = [0, 0, 0, 1]  # Background as black
        cmap_labels = ListedColormap(colors)
        
        axes[idx, 1].imshow(label_slice, cmap=cmap_labels)
        axes[idx, 1].set_title(f'Labels - Slice {slice_idx}')
        axes[idx, 1].axis('off')
        
        # Overlay
        axes[idx, 2].imshow(image_slice, cmap='gray', alpha=0.7)
        overlay_mask = label_slice > 0
        if np.any(overlay_mask):
            axes[idx, 2].imshow(np.ma.masked_where(~overlay_mask, label_slice), 
                              cmap=cmap_labels, alpha=0.5)
        axes[idx, 2].set_title(f'Overlay - Slice {slice_idx} ({annotation_pct:.1f}%)')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File paths
    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT3/RT023_left_CT3.nii.gz"
    label_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/label/CT3/RT023_left_CT3.nii.gz"
    # pred_path = "/home/user/auto-annotation/auto-annotation/predictions/RG018_right_CT2_prediction.nii.gz"
    # label_path = pred_path
    model_path = "/home/user/auto-annotation/auto-annotation/models/unetr/combined_model.pth"
    
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

        
        # Perform comprehensive analysis
        analysis_results = comprehensive_image_analysis(image, label, "comprehensive_analysis.txt")
        
        print(f"\nDetailed analysis saved to: comprehensive_analysis.txt")
        
        # Show some sample slices for visual verification
        show_sample_slices(image, label, num_samples=3)
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        import traceback
        traceback.print_exc()