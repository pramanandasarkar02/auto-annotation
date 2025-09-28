import os
import SimpleITK as sitk
import nrrd
import numpy as np
import nibabel as nib
import logging
import glob
import pandas as pd

def convert_nrrd_series(nrrd_path, output_path):
    data, header = nrrd.read(nrrd_path)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, output_path)
    logging.info(f"Converted {os.path.basename(nrrd_path)} to {os.path.basename(output_path)}")


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def extract_segment_info(header):
    """
    Extract segment information from NRRD header
    
    Args:
        header: NRRD header dictionary
        
    Returns:
        Dictionary mapping segment numbers to their info
    """
    segments = {}
    
    for key in header.keys():
        if key.startswith('Segment') and '_' in key and not key.startswith('Segmentation_'):
            parts = key.split('_', 1)
            if len(parts) == 2:
                try:
                    seg_prefix = parts[0]
                    if seg_prefix.startswith('Segment'):
                        seg_num_str = seg_prefix[7:]  # Remove 'Segment' prefix
                        if seg_num_str.isdigit():
                            seg_num = int(seg_num_str)
                            attr_name = parts[1]
                            
                            if seg_num not in segments:
                                segments[seg_num] = {}
                            segments[seg_num][attr_name] = header[key]
                except (ValueError, IndexError):
                    continue
    
    return segments

def get_spacing_from_header(header):
    """
    Extract voxel spacing from NRRD header
    
    Args:
        header: NRRD header dictionary
        
    Returns:
        tuple: (spacing_x, spacing_y, spacing_z) in mm
    """
    try:
        if 'space directions' in header:
            directions = header['space directions']
            spacing = []
            
            # Skip first direction if it's None (for the segment dimension)
            spatial_directions = [d for d in directions if d is not None]
            
            for direction in spatial_directions:
                if hasattr(direction, '__len__') and len(direction) >= 3:
                    # Calculate magnitude of direction vector
                    spacing.append(np.linalg.norm(direction))

            if len(spacing) >= 3:
                return tuple(spacing[-3:])
    
    except Exception as e:
        print(f"Warning: Could not extract spacing from header: {e}")
    
    # Default spacing if extraction fails
    return (1.0, 1.0, 1.0)

def assign_labels_bone_only(segments):
    """
    Assign specific labels for Bone and Wrong Bone only, nullify others
    Background - Label 0 (handled separately)
    Bone - Label 1
    Wrong Bone - Label 2
    All other segments - Nullified (background = 0)
    
    Args:
        segments: Dictionary of segment information
        
    Returns:
        Dictionary mapping segment numbers to their final labels and temp labels
    """
    segment_labels = {}
    logger = logging.getLogger(__name__)
    
    # First pass: assign temporary labels to avoid conflicts
    temp_label_start = 10
    temp_label_counter = 0
    
    logger.info("BONE-ONLY SEGMENTATION: Only keeping Bone and Wrong Bone segments")
    
    for seg_num, seg_info in segments.items():
        name = seg_info.get('Name', '').strip()
        name_lower = name.lower()
        
        # Assign temporary labels first (10, 11, 12, 13, 14, etc.)
        temp_label = temp_label_start + temp_label_counter
        temp_label_counter += 1
        
        # Determine final label - ONLY keep Bone and Wrong Bone
        if name_lower == 'bone' or ('bone' in name_lower and 'wrong' not in name_lower):
            final_label = 1
            logger.info(f"✓ KEEPING: Segment {seg_num} ('{name}') -> Temp Label {temp_label} -> Final Label 1 (Bone)")
        elif name_lower == 'wrong bone' or ('wrong' in name_lower and 'bone' in name_lower):
            final_label = 2
            logger.info(f"✓ KEEPING: Segment {seg_num} ('{name}') -> Temp Label {temp_label} -> Final Label 2 (Wrong Bone)")
        else:
            # NULLIFY all other segments (assign to background)
            final_label = 0
            logger.info(f"✗ NULLIFYING: Segment {seg_num} ('{name}') -> Temp Label {temp_label} -> Final Label 0 (Background - REMOVED)")
        
        segment_labels[seg_num] = {
            'temp_label': temp_label,
            'final_label': final_label,
            'name': name,
            'kept': final_label > 0  # Track which segments are kept
        }
    
    return segment_labels

def safe_label_assignment_bone_only(data, segment_info, segment_labels, logger):
    """
    Perform safe two-step label assignment keeping only Bone and Wrong Bone
    All other segments are nullified (set to background = 0)
    
    Args:
        data: The segmentation data array (4D: segments × Z × Y × X)
        segment_info: Dictionary with segment information
        segment_labels: Dictionary with temp and final label mappings
        logger: Logger instance
        
    Returns:
        Array with final labels assigned (3D: Z × Y × X)
    """
    logger.info("Starting safe two-step label assignment process for BONE-ONLY segmentation...")
    logger.info("BONE-ONLY MODE: Only Bone and Wrong Bone will be preserved, all others nullified")
    
    # Initialize output volume with background (0) - use 3D shape
    combined = np.zeros(data.shape[1:], dtype=np.uint8)
    
    # Step 1: Assign temporary labels by separating segments within each layer
    logger.info("Step 1: Separating and assigning temporary labels...")
    temp_combined = np.zeros(data.shape[1:], dtype=np.uint16)  # Use uint16 for temp labels
    
    logger.info(f"Data shape: {data.shape}, Total layers available: {data.shape[0]}")
    
    # Analyze the layer structure first
    for layer_idx in range(data.shape[0]):
        unique_values = np.unique(data[layer_idx])
        unique_values = unique_values[unique_values > 0]  # Remove background
        logger.info(f"Layer {layer_idx} contains label values: {unique_values}")
        for val in unique_values:
            voxel_count = np.sum(data[layer_idx] == val)
            logger.info(f"  Label value {val}: {voxel_count} voxels")
    
    segment_to_label_value = {}
    
    for seg_num in sorted(segment_info.keys()):
        layer = segment_info[seg_num]['layer']
        temp_label = segment_labels[seg_num]['temp_label']
        final_label = segment_labels[seg_num]['final_label']
        name = segment_labels[seg_num]['name']
        is_kept = segment_labels[seg_num]['kept']
        
        if is_kept:
            logger.info(f"✓ PROCESSING (KEEP): Segment {seg_num} ('{name}') on Layer {layer}")
        else:
            logger.info(f"✗ PROCESSING (NULLIFY): Segment {seg_num} ('{name}') on Layer {layer}")
        
        if layer < data.shape[0]:
            unique_values = np.unique(data[layer])
            unique_values = unique_values[unique_values > 0]  # Remove background
            
            # Determine which label value this segment should extract
            if layer == 0:  # Layer 0 contains 'Bone' and 'extra part'
                if name.lower() in ['bone'] or ('bone' in name.lower() and 'wrong' not in name.lower()):
                    target_label_value = 1  # Usually the first non-background label
                elif name.lower() in ['extra part']:
                    target_label_value = 2  # Usually the second label
                else:
                    # Fallback: assign based on segment order in layer
                    segments_in_layer = [s for s in segment_info.keys() if segment_info[s]['layer'] == layer]
                    segments_in_layer.sort()
                    target_label_value = segments_in_layer.index(seg_num) + 1
            
            elif layer == 1:  # Layer 1 contains 'Wrong bone' and 'Muscle'  
                if name.lower() in ['wrong bone'] or ('wrong' in name.lower() and 'bone' in name.lower()):
                    target_label_value = 1  # Usually the first non-background label
                elif name.lower() in ['muscle']:
                    target_label_value = 2  # Usually the second label
                else:
                    # Fallback: assign based on segment order in layer
                    segments_in_layer = [s for s in segment_info.keys() if segment_info[s]['layer'] == layer]
                    segments_in_layer.sort()
                    target_label_value = segments_in_layer.index(seg_num) + 1
            else:
                # For other layers, use first available label value
                target_label_value = unique_values[0] if len(unique_values) > 0 else 1
            
            segment_to_label_value[seg_num] = target_label_value
            
            # Extract voxels with the specific label value
            if target_label_value in unique_values:
                segment_mask = data[layer] == target_label_value
                original_voxels = np.sum(segment_mask)
                
                if is_kept:
                    logger.info(f"  ✓ EXTRACTING (KEEP): label value {target_label_value}: {original_voxels} voxels")
                else:
                    logger.info(f"  ✗ EXTRACTING (NULLIFY): label value {target_label_value}: {original_voxels} voxels -> will be set to background")
                
                if original_voxels > 0:
                    # Check for overlapping voxels with existing temp labels
                    overlap_mask = temp_combined[segment_mask] > 0
                    if np.any(overlap_mask):
                        overlapping_voxels = np.sum(overlap_mask)
                        if is_kept:
                            logger.warning(f"    ✓ Segment {seg_num} ('{name}'): {overlapping_voxels} voxels overlap with existing segments")
                        else:
                            logger.info(f"    ✗ Segment {seg_num} ('{name}'): {overlapping_voxels} voxels overlap (will be nullified anyway)")
                    
                    # Assign temporary label to voxels with this specific label value
                    # Even nullified segments get temp labels initially for proper processing
                    temp_combined[segment_mask] = temp_label
                    final_assigned = np.sum(temp_combined == temp_label)
                    
                    if is_kept:
                        logger.info(f"    ✓ Segment {seg_num} ('{name}'): {original_voxels} voxels extracted -> Temp Label {temp_label} (WILL KEEP)")
                    else:
                        logger.info(f"    ✗ Segment {seg_num} ('{name}'): {original_voxels} voxels extracted -> Temp Label {temp_label} (WILL NULLIFY)")
                else:
                    logger.warning(f"    Segment {seg_num} ('{name}'): No voxels found with label value {target_label_value}")
            else:
                logger.warning(f"    Segment {seg_num} ('{name}'): Label value {target_label_value} not found in layer {layer}")
                logger.warning(f"    Available label values: {unique_values}")
        else:
            logger.warning(f"    Segment {seg_num}: Layer {layer} out of range (max layer: {data.shape[0]-1})")
    
    # Debug: Check what temp labels actually exist
    unique_temp_labels = np.unique(temp_combined)
    logger.info(f"Temp labels successfully assigned: {unique_temp_labels[unique_temp_labels > 0]}")
    
    # Show the extraction summary
    logger.info(f"\nSEGMENT EXTRACTION SUMMARY (BONE-ONLY MODE):")
    kept_segments = 0
    nullified_segments = 0
    
    for seg_num, label_val in segment_to_label_value.items():
        temp_label = segment_labels[seg_num]['temp_label']
        final_label = segment_labels[seg_num]['final_label']
        name = segment_labels[seg_num]['name']
        layer = segment_info[seg_num]['layer']
        is_kept = segment_labels[seg_num]['kept']
        voxel_count = np.sum(temp_combined == temp_label)
        
        if is_kept:
            logger.info(f"   KEEP: Segment {seg_num} ('{name}') from Layer {layer}, Label Value {label_val} -> Temp {temp_label} -> Final {final_label}: {voxel_count} voxels")
            kept_segments += 1
        else:
            logger.info(f"   NULLIFY: Segment {seg_num} ('{name}') from Layer {layer}, Label Value {label_val} -> Temp {temp_label} -> Final {final_label} (Background): {voxel_count} voxels")
            nullified_segments += 1
    
    logger.info(f"\nSUMMARY: {kept_segments} segments kept, {nullified_segments} segments nullified")
    
    # Step 2: Convert temporary labels to final labels (0, 1, 2 only)
    logger.info("\nStep 2: Converting temporary labels to final labels (BONE-ONLY MODE)...")
    
    # Apply final label mapping - process ALL temp labels
    for seg_num in sorted(segment_labels.keys()):
        temp_label = segment_labels[seg_num]['temp_label']
        final_label = segment_labels[seg_num]['final_label']
        seg_name = segment_labels[seg_num]['name']
        is_kept = segment_labels[seg_num]['kept']
        
        temp_mask = temp_combined == temp_label
        voxel_count = np.sum(temp_mask)
        
        if voxel_count > 0:
            combined[temp_mask] = final_label
            if is_kept:
                logger.info(f"     KEEP: Temp Label {temp_label} -> Final Label {final_label} ({seg_name}): {voxel_count} voxels")
            else:
                logger.info(f"     NULLIFY: Temp Label {temp_label} -> Final Label {final_label} ({seg_name}): {voxel_count} voxels -> Background")
        else:
            if is_kept:
                logger.warning(f"    KEEP: Temp Label {temp_label} -> Final Label {final_label} ({seg_name}): 0 voxels (segment extraction failed)")
            else:
                logger.info(f"     NULLIFY: Temp Label {temp_label} -> Final Label {final_label} ({seg_name}): 0 voxels (already background)")
    
    # Final verification
    final_unique_labels = np.unique(combined)
    logger.info(f"\nFINAL BONE-ONLY SEGMENTATION RESULT:")
    logger.info(f"   Final labels present: {final_unique_labels}")
    logger.info(f"   Expected labels: [0, 1, 2] (Background, Bone, Wrong Bone)")
    
    expected_labels = set([0, 1, 2])
    missing_labels = expected_labels - set(final_unique_labels)
    unexpected_labels = set(final_unique_labels) - expected_labels
    
    if missing_labels:
        logger.warning(f"Missing expected labels: {missing_labels}")
    if unexpected_labels:
        logger.warning(f"Unexpected labels found: {unexpected_labels}")
    
    if final_unique_labels.tolist() == [0] or set(final_unique_labels).issubset({0, 1, 2}):
        logger.info(f"   ✓ Bone-only segmentation successful! Only bone segments preserved.")
    
    return combined

def convert_nrrd_to_nifti_bone_only(nrrd_path, output_path):
    """
    NRRD to NIfTI conversion keeping only Bone and Wrong Bone segments
    
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
        
        # Assign labels for bone-only processing
        segment_labels = assign_labels_bone_only(segments)
        
        # Get voxel spacing for volume calculations
        spacing = get_spacing_from_header(header)
        voxel_volume_mm3 = np.prod(spacing)
        logger.info(f"Voxel spacing: {spacing} mm, Volume per voxel: {voxel_volume_mm3:.6f} mm³")
        
        # Print segment details with assigned labels
        segment_info = {}
        logger.info("\nSegment to Label Mapping (BONE-ONLY MODE):")

        # STEP 1: First loop - Assign all segments to temporary labels
        logger.info("STEP 1: Assigning temporary labels...")
        for seg_num in sorted(segments.keys()):
            seg_info = segments[seg_num]
            name = seg_info.get('Name', f'Segment_{seg_num}')
            layer = int(seg_info.get('Layer', 0))
            color = seg_info.get('Color', 'N/A')
            temp_label = segment_labels[seg_num]['temp_label']
            is_kept = segment_labels[seg_num]['kept']
            
            segment_info[seg_num] = {
                'name': name,
                'layer': layer,
                'temp_label': temp_label,
                'final_label': None,  # Will be assigned in step 2
                'color': color,
                'kept': is_kept
            }
            
            if is_kept:
                logger.info(f"  ✓ KEEP: Segment {seg_num}: '{name}' -> Temp Label {temp_label} (Layer {layer})")
            else:
                logger.info(f"  ✗ NULLIFY: Segment {seg_num}: '{name}' -> Temp Label {temp_label} (Layer {layer})")

        logger.info(" All segments assigned temporary labels")

        # STEP 2: Second loop - Convert temporary labels to final labels
        logger.info("\nSTEP 2: Converting temporary labels to final labels...")
        for seg_num in sorted(segments.keys()):
            final_label = segment_labels[seg_num]['final_label']
            temp_label = segment_info[seg_num]['temp_label']
            name = segment_info[seg_num]['name']
            is_kept = segment_info[seg_num]['kept']
            
            # Update the segment_info with final label
            segment_info[seg_num]['final_label'] = final_label
            
            if is_kept:
                logger.info(f"  ✓ KEEP: Segment {seg_num}: '{name}' -> Temp {temp_label} -> Final Label {final_label}")
            else:
                logger.info(f"  ✗ NULLIFY: Segment {seg_num}: '{name}' -> Temp {temp_label} -> Final Label {final_label} (Background)")

        logger.info(" All temporary labels converted to final labels")

        logger.info(f"\n📋 COMPLETE MAPPING SUMMARY (BONE-ONLY):")
        for seg_num in sorted(segment_info.keys()):
            info = segment_info[seg_num]
            status = "KEEP" if info['kept'] else "NULLIFY"
            logger.info(f"  {status}: Segment {seg_num}: '{info['name']}' | Temp: {info['temp_label']} | Final: {info['final_label']} | Layer: {info['layer']}")
        
        # Process data based on dimensions
        if data.ndim == 4:
            logger.info("Processing 4D data (segments x Z x Y x X)")
            
            # Use bone-only safe label assignment process
            combined = safe_label_assignment_bone_only(data, segment_info, segment_labels, logger)
            
            data = combined
            
        elif data.ndim == 3:
            logger.info("Processing 3D data")
            # For 3D data, assume it's already a label map - apply remapping if needed
            logger.warning("3D data detected - manual label remapping may be required")
            
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
        
        # Analyze final label distribution
        unique_labels, counts = np.unique(data, return_counts=True)
        logger.info("\nFinal label distribution after bone-only assignment:")
        
        # Define label names for reporting (bone-only)
        label_names = {
            0: "Background",
            1: "Bone", 
            2: "Wrong Bone"
        }
        
        volume_info = []
        total_volume_mm3 = 0
        
        for label, count in zip(unique_labels, counts):
            volume_mm3 = count * voxel_volume_mm3
            
            # Get name from predefined mapping or segment info
            if label in label_names:
                name = label_names[label]
            else:
                name = "Unknown"
                for seg_num, info in segment_info.items():
                    if info['final_label'] == label:
                        name = info['name']
                        break
            
            volume_info.append({
                'label': int(label),
                'name': name,
                'voxel_count': int(count),
                'volume_mm3': float(volume_mm3),
                'percentage': float(count / data.size * 100)
            })
            
            if label > 0:  # Don't count background in total
                total_volume_mm3 += volume_mm3
            
            logger.info(f"  Label {label} ({name}): {count} voxels, "
                       f"{volume_mm3:.2f} mm³ ({count/data.size*100:.2f}%)")
        
        logger.info(f"\nTotal bone volume: {total_volume_mm3:.2f} mm³")
        
        # Convert to SimpleITK image
        img = sitk.GetImageFromArray(data)
        
        # Set spacing and origin information
        try:
            img.SetSpacing(spacing[::-1])  # SimpleITK expects reversed order
            logger.info(f"Set spacing: {spacing[::-1]}")
            
            if 'space origin' in header:
                origin = header['space origin']
                if hasattr(origin, '__len__') and len(origin) >= 3:
                    img.SetOrigin(list(reversed(origin[:3])))
                    logger.info(f"Set origin: {list(reversed(origin[:3]))}")
                    
        except Exception as e:
            logger.warning(f"Could not set spacing/origin info: {e}")
        
        # Write NIfTI file
        sitk.WriteImage(img, output_path)
        logger.info(f"Successfully saved: {output_path}")
        
        return {
            'segments': segment_info,
            'volume_info': volume_info,
            'total_volume_mm3': total_volume_mm3,
            'voxel_volume_mm3': voxel_volume_mm3,
            'spacing_mm': spacing,
            'segment_labels': segment_labels
        }
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

def create_volume_report_bone_only(results, output_dir):
    """
    Create detailed volume report for bone-only segmentation
    
    Args:
        results: Results from conversion
        output_dir: Directory to save reports
    """
    logger = logging.getLogger(__name__)
    
    # Create DataFrame for volume information
    volume_df = pd.DataFrame(results['volume_info'])
    
    # Save volume report
    volume_csv = os.path.join(output_dir, "volume_analysis_bone_only.csv")
    volume_df.to_csv(volume_csv, index=False)
    logger.info(f"Volume analysis saved to: {volume_csv}")
    
    # Create segment summary with bone-only labeling info
    segment_df = pd.DataFrame([
        {
            'segment_id': seg_id,
            'segment_name': info['name'],
            'temp_label': info.get('temp_label', 'N/A'),
            'final_label': info.get('final_label', info.get('label', 'N/A')),
            'layer': info['layer'],
            'color': info['color'],
            'status': 'KEPT' if info.get('kept', False) else 'NULLIFIED'
        }
        for seg_id, info in results['segments'].items()
    ])
    
    segment_csv = os.path.join(output_dir, "segment_mapping_bone_only.csv")
    segment_df.to_csv(segment_csv, index=False)
    logger.info(f"Segment mapping saved to: {segment_csv}")
    
    # Create summary report
    kept_segments = sum(1 for info in results['segments'].values() if info.get('kept', False))
    nullified_segments = len(results['segments']) - kept_segments
    
    report_lines = [
        "NRRD BONE-ONLY SEGMENTATION ANALYSIS REPORT",
        "=" * 70,
        f"Total original segments: {len(results['segments'])}",
        f"Segments kept: {kept_segments}",
        f"Segments nullified: {nullified_segments}",
        f"Voxel spacing: {results['spacing_mm']} mm",
        f"Volume per voxel: {results['voxel_volume_mm3']:.6f} mm³",
        f"Total bone volume: {results['total_volume_mm3']:.2f} mm³",
        "",
        "BONE-ONLY PROCESSING:",
        "-" * 50,
        "Only 'Bone' and 'Wrong Bone' segments are preserved.",
        "All other segments (Muscle, Extra Part, etc.) are nullified to background.",
        "Step 1: All segments assigned temporary labels (10, 11, 12, 13, 14, ...)",
        "Step 2: Only bone segments get final labels (1, 2), others get 0 (background)",
        "This ensures clean bone-only segmentation for analysis.",
        "",
        "BONE-ONLY LABEL MAPPING:",
        "-" * 40,
        "Background → Label 0 (includes nullified segments)",
        "Bone → Label 1",
        "Wrong Bone → Label 2",
        "",
        "SEGMENT PROCESSING RESULTS:",
        "-" * 30
    ]
    
    # Show segment to label mapping with status
    for seg_id, info in sorted(results['segments'].items()):
        temp_label = info.get('temp_label', 'N/A')
        final_label = info.get('final_label', info.get('label', 'N/A'))
        status = "✓ KEPT" if info.get('kept', False) else "✗ NULLIFIED"
        report_lines.append(f"{status}: Segment {seg_id}: '{info['name']}' → Temp {temp_label} → Final {final_label}")
    
    report_lines.extend([
        "",
        "FINAL VOLUME DETAILS:",
        "-" * 30
    ])
    
    for info in results['volume_info']:
        if info['label'] == 0:
            report_lines.append(
                f"Label {info['label']}: {info['name']} (includes nullified segments)\n"
                f"  Volume: {info['volume_mm3']:.2f} mm³\n"
                f"  Voxels: {info['voxel_count']:,}\n"
                f"  Percentage: {info['percentage']:.2f}%\n"
            )
        elif info['label'] > 0:  # Only bone segments
            report_lines.append(
                f"Label {info['label']}: {info['name']}\n"
                f"  Volume: {info['volume_mm3']:.2f} mm³\n"
                f"  Voxels: {info['voxel_count']:,}\n"
                f"  Percentage: {info['percentage']:.2f}%\n"
            )
    
    report_txt = os.path.join(output_dir, "bone_only_segmentation_report.txt")
    with open(report_txt, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Summary report saved to: {report_txt}")
    
    return volume_csv, segment_csv, report_txt


def convert_nrrd_to_nifti(nrrd_dir, output_dir):
    for patient in os.listdir(nrrd_dir):
        patient_path = os.path.join(nrrd_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        for series in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series)
            if not os.path.isdir(series_path):
                logging.warning(f"Skipping non-directory file: {os.path.basename(series_path)}")
                continue
            nrrd_path = os.path.join(series_path, f"Segmentation.seg.nrrd")
            
            output_path = os.path.join(output_dir, series.upper())
            if not os.path.exists(nrrd_path):
                logging.warning(f"Skipping series without nrrd file: {os.path.basename(nrrd_path)}")