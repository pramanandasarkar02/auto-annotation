import nrrd
import SimpleITK as sitk
import numpy as np
import os
import logging
import pandas as pd

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def extract_segment_info(header):
    """
    Extract segment information from NRRD header in a robust way
    
    Args:
        header: NRRD header dictionary
        
    Returns:
        Dictionary mapping segment numbers to their info
    """
    segments = {}
    
    # Find segment keys (exclude Segmentation_ keys)
    for key in header.keys():
        if key.startswith('Segment') and '_' in key and not key.startswith('Segmentation_'):
            parts = key.split('_', 1)
            if len(parts) == 2:
                try:
                    # Extract segment number
                    seg_prefix = parts[0]
                    if seg_prefix.startswith('Segment'):
                        seg_num_str = seg_prefix[7:]  # Remove 'Segment'
                        if seg_num_str.isdigit():
                            seg_num = int(seg_num_str)
                            attr_name = parts[1]
                            
                            if seg_num not in segments:
                                segments[seg_num] = {}
                            segments[seg_num][attr_name] = header[key]
                except (ValueError, IndexError):
                    continue
    
    return segments

def convert_nrrd_to_nifti_clean(nrrd_path, output_path):
    """
    Clean NRRD to NIfTI conversion focusing on proper label assignment
    
    Args:
        nrrd_path: Path to input NRRD file
        output_path: Path to output NIfTI file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Read NRRD file
        data, header = nrrd.read(nrrd_path)
        logger.info(f"Loaded NRRD with shape: {data.shape}")
        
        # Extract segment information
        segments = extract_segment_info(header)
        logger.info(f"Found {len(segments)} segments")
        
        # Print segment details
        for seg_num in sorted(segments.keys()):
            seg_info = segments[seg_num]
            name = seg_info.get('Name', f'Segment_{seg_num}')
            label_value = seg_info.get('LabelValue', seg_num)
            layer = seg_info.get('Layer', 0)
            logger.info(f"  Segment {seg_num}: '{name}' -> Label {label_value} (Layer {layer})")
        
        # Handle 4D data by combining segments
        if data.ndim == 4:
            logger.info("Processing 4D data - combining segments...")
            combined = np.zeros(data.shape[1:], dtype=np.uint8)
            
            # Strategy: Use segment metadata to properly assign labels
            for seg_num in sorted(segments.keys()):
                seg_info = segments[seg_num]
                try:
                    layer = int(seg_info.get('Layer', 0))
                    label_value = int(seg_info.get('LabelValue', seg_num))
                    
                    # Make sure layer index is valid
                    if layer < data.shape[0]:
                        # Get mask for this segment
                        segment_mask = data[layer] > 0
                        voxel_count = np.sum(segment_mask)
                        
                        if voxel_count > 0:
                            # Assign label value to these voxels
                            combined[segment_mask] = label_value
                            logger.info(f"    Assigned {voxel_count} voxels to label {label_value}")
                        else:
                            logger.warning(f"    No voxels found for segment {seg_num}")
                    else:
                        logger.warning(f"    Layer {layer} out of range for segment {seg_num}")
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"    Error processing segment {seg_num}: {e}")
                    continue
            
            data = combined
        
        elif data.ndim == 3:
            logger.info("Processing 3D data")
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
        
        # Check final label distribution
        unique_labels, counts = np.unique(data, return_counts=True)
        logger.info("Final label distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Label {label}: {count} voxels")
        
        # Convert to SimpleITK image
        img = sitk.GetImageFromArray(data)
        
        # Set spacing information if available
        try:
            if 'space directions' in header:
                spacing = []
                directions = header['space directions']
                for direction in directions:
                    if direction is not None and hasattr(direction, '__len__'):
                        spacing.append(np.linalg.norm(direction))
                
                if spacing:
                    # Reverse for ITK convention (z,y,x -> x,y,z)
                    img.SetSpacing(spacing[::-1])
                    logger.info(f"Set spacing: {spacing[::-1]}")
            
            if 'space origin' in header:
                origin = header['space origin']
                if hasattr(origin, '__len__'):
                    # Reverse for ITK convention
                    img.SetOrigin(list(reversed(origin)))
                    logger.info(f"Set origin: {list(reversed(origin))}")
                    
        except Exception as e:
            logger.warning(f"Could not set spacing/origin info: {e}")
        
        # Write NIfTI file
        sitk.WriteImage(img, output_path)
        logger.info(f"Successfully saved: {output_path}")
        
        return {
            'segments': segments,
            'final_labels': unique_labels.tolist(),
            'label_counts': counts.tolist()
        }
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

def analyze_nifti_labels(nifti_path):
    """
    Analyze the labels in the converted NIfTI file
    
    Args:
        nifti_path: Path to NIfTI file
        
    Returns:
        Analysis results
    """
    logger = logging.getLogger(__name__)
    
    try:
        img = sitk.ReadImage(nifti_path)
        data = sitk.GetArrayFromImage(img)
        
        # Get spacing for volume calculations
        spacing = img.GetSpacing()
        voxel_volume = np.prod(spacing)  # mm³ per voxel
        
        # Analyze labels
        unique_labels, counts = np.unique(data, return_counts=True)
        total_voxels = data.size
        
        results = []
        for label, count in zip(unique_labels, counts):
            volume_mm3 = count * voxel_volume
            percentage = (count / total_voxels) * 100
            
            results.append({
                'label': int(label),
                'voxel_count': int(count),
                'volume_mm3': float(volume_mm3),
                'percentage': float(percentage)
            })
            
            logger.info(f"Label {label}: {count} voxels, {volume_mm3:.2f} mm³, {percentage:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def main():
    """Main processing function"""
    logger = setup_logging()
    
    # Configuration
    input_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/mask/RT023_left/ct2/Segmentation.seg.nrrd"
    output_dir = "./"
    output_nifti = os.path.join(output_dir, "segmentation_clean.nii.gz")
    
    try:
        logger.info("Starting NRRD to NIfTI conversion...")
        
        # Convert NRRD to NIfTI
        conversion_result = convert_nrrd_to_nifti_clean(input_path, output_nifti)
        
        # Analyze the result
        logger.info("\nAnalyzing converted file...")
        analysis_result = analyze_nifti_labels(output_nifti)
        
        # Create summary DataFrame
        df = pd.DataFrame(analysis_result)
        csv_path = os.path.join(output_dir, "label_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        # Print summary
        print("\n" + "="*50)
        print("CONVERSION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Input file: {os.path.basename(input_path)}")
        print(f"Output file: {output_nifti}")
        print(f"Analysis saved: {csv_path}")
        print(f"\nFound {len(analysis_result)} unique labels (including background):")
        
        # Show background
        bg_result = next((r for r in analysis_result if r['label'] == 0), None)
        if bg_result:
            print(f"  Label 0 (Background): {bg_result['voxel_count']} voxels "
                  f"({bg_result['volume_mm3']:.1f} mm³, {bg_result['percentage']:.1f}%)")
        
        # Show each segment with its name
        segment_names = {0: 'Bone', 1: 'Wrong bone', 2: 'Muscle', 3: 'extra part'}
        for result in analysis_result:
            if result['label'] != 0:  # Skip background
                seg_num = result['label'] - 1  # Convert back to segment number
                seg_name = segment_names.get(seg_num, f'Segment_{seg_num}')
                print(f"  Label {result['label']} ({seg_name}): {result['voxel_count']} voxels "
                      f"({result['volume_mm3']:.1f} mm³, {result['percentage']:.1f}%)")
        
        print("\nSegmentation file ready for further processing!")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()