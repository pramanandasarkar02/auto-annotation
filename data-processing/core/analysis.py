import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def deep_analyze_nrrd_structure(nrrd_path):
    """
    Perform deep analysis of NRRD file to understand the segmentation structure
    """
    logger = logging.getLogger(__name__)
    logger.info(f"="*80)
    logger.info(f"DEEP ANALYSIS: {os.path.basename(nrrd_path)}")
    logger.info(f"="*80)
    
    try:
        data, header = nrrd.read(nrrd_path)
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Data type: {data.dtype}")
        logger.info(f"Data range: {np.min(data)} to {np.max(data)}")
        
        # Analyze header for segment information
        logger.info("\n--- HEADER ANALYSIS ---")
        segment_keys = [k for k in header.keys() if 'segment' in k.lower()]
        logger.info(f"Found {len(segment_keys)} segment-related keys")
        
        segments = {}
        for key in sorted(segment_keys):
            logger.info(f"  {key}: {header[key]}")
            
            # Extract segment info
            if key.startswith('Segment') and '_' in key:
                parts = key.split('_', 1)
                if len(parts) == 2:
                    seg_prefix = parts[0]
                    if seg_prefix.startswith('Segment'):
                        seg_num_str = seg_prefix[7:]
                        if seg_num_str.isdigit():
                            seg_num = int(seg_num_str)
                            attr_name = parts[1]
                            if seg_num not in segments:
                                segments[seg_num] = {}
                            segments[seg_num][attr_name] = header[key]
        
        logger.info(f"\nExtracted {len(segments)} segments:")
        for seg_num in sorted(segments.keys()):
            seg_info = segments[seg_num]
            name = seg_info.get('Name', f'Unknown_{seg_num}')
            layer = seg_info.get('Layer', 'Unknown')
            color = seg_info.get('Color', 'Unknown')
            logger.info(f"  Segment {seg_num}: Name='{name}', Layer={layer}, Color={color}")
        
        # Analyze data structure
        logger.info("\n--- DATA STRUCTURE ANALYSIS ---")
        if data.ndim == 4:
            logger.info("4D Data - analyzing each layer:")
            layer_stats = {}
            
            for layer in range(data.shape[0]):
                layer_data = data[layer]
                unique_vals, counts = np.unique(layer_data, return_counts=True)
                nonzero_count = np.sum(layer_data > 0)
                total_voxels = layer_data.size
                
                layer_stats[layer] = {
                    'unique_values': unique_vals,
                    'counts': counts,
                    'nonzero_voxels': nonzero_count,
                    'nonzero_percentage': (nonzero_count / total_voxels) * 100
                }
                
                logger.info(f"  Layer {layer}:")
                logger.info(f"    Shape: {layer_data.shape}")
                logger.info(f"    Unique values: {unique_vals}")
                logger.info(f"    Value counts: {counts}")
                logger.info(f"    Non-zero voxels: {nonzero_count:,} ({(nonzero_count/total_voxels)*100:.2f}%)")
                
                # Check if this layer corresponds to a segment
                layer_segments = [seg for seg, info in segments.items() 
                                if str(info.get('Layer', '')) == str(layer)]
                if layer_segments:
                    logger.info(f"    Associated segments: {layer_segments}")
                    for seg_num in layer_segments:
                        seg_name = segments[seg_num].get('Name', f'Segment_{seg_num}')
                        logger.info(f"      Segment {seg_num}: '{seg_name}'")
        
        elif data.ndim == 3:
            logger.info("3D Data - single volume analysis:")
            unique_vals, counts = np.unique(data, return_counts=True)
            total_voxels = data.size
            
            logger.info(f"  Shape: {data.shape}")
            logger.info(f"  Unique values: {unique_vals}")
            logger.info(f"  Value distribution:")
            for val, count in zip(unique_vals, counts):
                percentage = (count / total_voxels) * 100
                logger.info(f"    Value {val}: {count:,} voxels ({percentage:.2f}%)")
        
        # Analyze spatial distribution
        logger.info("\n--- SPATIAL DISTRIBUTION ANALYSIS ---")
        if data.ndim >= 3:
            # Get the 3D volume (last 3 dimensions)
            if data.ndim == 4:
                # Combine all layers into a single volume for spatial analysis
                combined_volume = np.max(data, axis=0)  # Take max across layers
            else:
                combined_volume = data
            
            # Find bounding box of all non-zero voxels
            nonzero_coords = np.where(combined_volume > 0)
            if len(nonzero_coords[0]) > 0:
                bbox = {
                    'z': (np.min(nonzero_coords[0]), np.max(nonzero_coords[0])),
                    'y': (np.min(nonzero_coords[1]), np.max(nonzero_coords[1])),
                    'x': (np.min(nonzero_coords[2]), np.max(nonzero_coords[2]))
                }
                logger.info(f"  Bounding box of all segmentations:")
                logger.info(f"    Z: {bbox['z'][0]} to {bbox['z'][1]} (span: {bbox['z'][1] - bbox['z'][0] + 1})")
                logger.info(f"    Y: {bbox['y'][0]} to {bbox['y'][1]} (span: {bbox['y'][1] - bbox['y'][0] + 1})")
                logger.info(f"    X: {bbox['x'][0]} to {bbox['x'][1]} (span: {bbox['x'][1] - bbox['x'][0] + 1})")
                
                # Calculate volume coverage
                total_span = (bbox['z'][1] - bbox['z'][0] + 1) * (bbox['y'][1] - bbox['y'][0] + 1) * (bbox['x'][1] - bbox['x'][0] + 1)
                actual_voxels = len(nonzero_coords[0])
                density = (actual_voxels / total_span) * 100
                logger.info(f"    Segmentation density in bounding box: {density:.2f}%")
            else:
                logger.warning("  No non-zero voxels found!")
        
        # Problem diagnosis
        logger.info("\n--- PROBLEM DIAGNOSIS ---")
        issues_found = []
        
        # Check for extremely sparse segmentations
        if data.ndim == 4:
            for layer in range(data.shape[0]):
                nonzero_pct = layer_stats[layer]['nonzero_percentage']
                if nonzero_pct < 1.0:  # Less than 1% of voxels are labeled
                    issues_found.append(f"Layer {layer} extremely sparse ({nonzero_pct:.3f}% labeled)")
        
        # Check for single-value layers
        if data.ndim == 4:
            for layer in range(data.shape[0]):
                unique_vals = layer_stats[layer]['unique_values']
                if len(unique_vals) == 1:
                    issues_found.append(f"Layer {layer} contains only background (value {unique_vals[0]})")
                elif len(unique_vals) == 2 and 0 in unique_vals:
                    issues_found.append(f"Layer {layer} binary segmentation (only background + 1 label)")
        
        # Check for inconsistent segment naming
        segment_names = [seg.get('Name', '') for seg in segments.values()]
        if len(set(segment_names)) != len(segment_names):
            issues_found.append("Duplicate segment names found")
        
        # Check for missing layer assignments
        assigned_layers = set()
        for seg_info in segments.values():
            layer = seg_info.get('Layer', None)
            if layer is not None:
                assigned_layers.add(int(layer))
        
        if data.ndim == 4:
            available_layers = set(range(data.shape[0]))
            unassigned_layers = available_layers - assigned_layers
            if unassigned_layers:
                issues_found.append(f"Layers without segment assignment: {sorted(unassigned_layers)}")
        
        if issues_found:
            logger.warning("ISSUES DETECTED:")
            for issue in issues_found:
                logger.warning(f"  ⚠ {issue}")
        else:
            logger.info("✓ No obvious structural issues detected")
        
        # Recommendations
        logger.info("\n--- RECOMMENDATIONS ---")
        
        if data.ndim == 4:
            total_labeled_voxels = sum(stats['nonzero_voxels'] for stats in layer_stats.values())
            total_voxels = data.size
            overall_density = (total_labeled_voxels / total_voxels) * 100
            
            if overall_density < 5:
                logger.info("  📌 VERY SPARSE SEGMENTATION detected:")
                logger.info("     - This explains the severe class imbalance")
                logger.info("     - Most voxels are naturally background")
                logger.info("     - Consider this is normal for precise anatomical segmentations")
            
            # Check if segments are properly distributed
            layer_densities = [stats['nonzero_percentage'] for stats in layer_stats.values()]
            if max(layer_densities) / min([d for d in layer_densities if d > 0]) > 10:
                logger.info("  📌 UNEVEN LAYER DISTRIBUTION:")
                logger.info("     - Some layers have much more segmentation than others")
                logger.info("     - This may be anatomically correct (different structure sizes)")
        
        logger.info("  📌 GENERAL RECOMMENDATIONS:")
        logger.info("     1. Verify segmentations are anatomically correct")
        logger.info("     2. Consider whether class imbalance is expected for your anatomy")
        logger.info("     3. Use weighted loss functions if training ML models")
        logger.info("     4. Consider focal loss or class balancing techniques")
        
        return {
            'data_shape': data.shape,
            'segments': segments,
            'layer_stats': layer_stats if data.ndim == 4 else None,
            'issues_found': issues_found,
            'total_labeled_percentage': (np.sum(data > 0) / data.size) * 100
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return None

def analyze_batch_nrrd_files(nrrd_dir):
    """
    Analyze all NRRD files in a directory to find patterns
    """
    logger = setup_logging()
    logger.info("BATCH NRRD ANALYSIS")
    logger.info("="*80)
    
    analysis_results = []
    
    for patient in os.listdir(nrrd_dir):
        patient_path = os.path.join(nrrd_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        
        for series in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series)
            if not os.path.isdir(series_path):
                continue
            
            nrrd_path = os.path.join(series_path, "Segmentation.seg.nrrd")
            if os.path.exists(nrrd_path):
                logger.info(f"\nAnalyzing: {patient}/{series}")
                result = deep_analyze_nrrd_structure(nrrd_path)
                if result:
                    result['patient'] = patient
                    result['series'] = series
                    result['file_path'] = nrrd_path
                    analysis_results.append(result)
    
    # Summary analysis
    if analysis_results:
        logger.info("\n" + "="*80)
        logger.info("BATCH SUMMARY")
        logger.info("="*80)
        
        total_files = len(analysis_results)
        logger.info(f"Total files analyzed: {total_files}")
        
        # Analyze common patterns
        all_issues = []
        labeled_percentages = []
        
        for result in analysis_results:
            all_issues.extend(result['issues_found'])
            labeled_percentages.append(result['total_labeled_percentage'])
        
        # Most common issues
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split('(')[0].strip()  # Get issue type without details
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        if issue_counts:
            logger.info("\nMost common issues:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {issue}: {count}/{total_files} files ({(count/total_files)*100:.1f}%)")
        
        # Labeled percentage statistics
        if labeled_percentages:
            avg_labeled = np.mean(labeled_percentages)
            min_labeled = np.min(labeled_percentages)
            max_labeled = np.max(labeled_percentages)
            
            logger.info(f"\nLabeled voxel statistics:")
            logger.info(f"  Average: {avg_labeled:.2f}% of voxels are labeled")
            logger.info(f"  Range: {min_labeled:.2f}% to {max_labeled:.2f}%")
            
            if avg_labeled < 5:
                logger.info("  ⚠ CONCLUSION: Segmentations are naturally very sparse!")
                logger.info("    This explains the severe class imbalance you're seeing.")
                logger.info("    This may be normal for precise anatomical segmentations.")
    
    return analysis_results

def quick_fix_recommendations():
    """
    Provide quick recommendations based on common segmentation issues
    """
    logger = setup_logging()
    logger.info("QUICK FIX RECOMMENDATIONS")
    logger.info("="*50)
    
    logger.info("If your segmentations are very sparse (< 5% labeled), this is likely NORMAL for:")
    logger.info("  • Bone segmentations (compact bone is small volume)")
    logger.info("  • Precise anatomical structures")
    logger.info("  • Medical imaging segmentations")
    
    logger.info("\nInstead of trying to 'balance' the labels, consider:")
    logger.info("  1. Use WEIGHTED LOSS FUNCTIONS in your ML models")
    logger.info("  2. Use FOCAL LOSS for handling class imbalance")
    logger.info("  3. Oversample minority classes during training")
    logger.info("  4. Use evaluation metrics appropriate for imbalanced data (F1, IoU, Dice)")
    
    logger.info("\nDO NOT try to artificially balance by:")
    logger.info("  ❌ Combining anatomically different structures")
    logger.info("  ❌ Losing label classes (as your 'balanced' version did)")
    logger.info("  ❌ Randomly reassigning voxels")
    
    logger.info("\nVerify your data with:")
    logger.info("  1. Visual inspection in 3D Slicer or ITK-SNAP")
    logger.info("  2. Check if segmentations match expected anatomy")
    logger.info("  3. Ensure all anatomical structures are preserved")

# Example usage functions
def analyze_single_file_detailed(nrrd_path):
    """Analyze a single NRRD file in detail"""
    return deep_analyze_nrrd_structure(nrrd_path)

def analyze_your_dataset(nrrd_directory):
    """Analyze your entire NRRD dataset"""
    return analyze_batch_nrrd_files(nrrd_directory)

if __name__ == "__main__":
    # Run this to get quick recommendations
    quick_fix_recommendations()
    
    # Then run this to analyze your data:
    # analyze_your_dataset("/path/to/your/nrrd_directory")
    # 
    # Or analyze a single file:
    analyze_single_file_detailed("/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/mask/RG018_left/ct2/Segmentation.seg.nrrd")