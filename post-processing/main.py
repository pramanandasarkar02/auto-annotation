import numpy as np
import pydicom
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes
from skimage import measure, morphology
from skimage.segmentation import clear_border
import os
import glob
from pathlib import Path

class FemurDICOMProcessor:
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.dicom_dataset = None
        self.volume_data = None
        self.dicom_files = []
        
    def load_dicom_directory(self, dicom_dir):
        """Load all DICOM files from directory"""
        try:
            dicom_path = Path(dicom_dir)
            # Find all DICOM files
            dicom_files = []
            for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
                dicom_files.extend(glob.glob(str(dicom_path / ext)))
            
            if not dicom_files:
                # Try files without extension
                for file in dicom_path.iterdir():
                    if file.is_file():
                        try:
                            pydicom.dcmread(str(file), stop_before_pixels=True, force=True)
                            dicom_files.append(str(file))
                        except:
                            continue

            
            if not dicom_files:
                print(f"No DICOM files found in {dicom_dir}")
                return False
            
            # Sort files by instance number or filename
            self.dicom_files = sorted(dicom_files)
            print(f"Found {len(self.dicom_files)} DICOM files")
            
            # Load first file to get metadata
            self.dicom_dataset = pydicom.dcmread(self.dicom_files[0], force=True)
            
            # Load all slices into volume
            volume_slices = []
            for file_path in self.dicom_files:
                ds = pydicom.dcmread(file_path, force=True)
                volume_slices.append(ds.pixel_array)
            
            self.volume_data = np.stack(volume_slices, axis=0).astype(np.float64)
            self.original_data = self.volume_data.copy()
            self.processed_data = self.volume_data.copy()
            
            print(f"Loaded volume: {self.volume_data.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading DICOM directory: {e}")
            return False
        
    def load_dicom(self, dicom_path):
        """Load DICOM file"""
        try:
            self.dicom_dataset = pydicom.dcmread(dicom_path, force=True)
            self.original_data = self.dicom_dataset.pixel_array.astype(np.float64)
            self.processed_data = self.original_data.copy()
            print(f"Loaded DICOM: {self.original_data.shape}")
            return True
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            return False
    
    def threshold_image(self, threshold=0.5):
        """Convert to binary mask using threshold"""
        if len(self.processed_data.shape) == 2:
            # Single slice
            self.processed_data = (self.processed_data > threshold).astype(np.uint8)
        else:
            # Volume
            self.processed_data = (self.processed_data > threshold).astype(np.uint8)
        print("Applied thresholding")
    
    def keep_largest_component(self, connectivity=None):
        """Keep only the largest connected component"""
        if len(self.processed_data.shape) == 2:
            # 2D case
            labeled = measure.label(self.processed_data, connectivity=connectivity)
        else:
            # 3D case
            labeled = measure.label(self.processed_data, connectivity=connectivity)
        
        if labeled.max() == 0:
            print("No connected components found")
            return
        
        # Find largest component
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # Ignore background
        largest_component = component_sizes.argmax()
        
        # Keep only largest component
        self.processed_data = (labeled == largest_component).astype(np.uint8)
        print(f"Kept largest component (size: {component_sizes[largest_component]} voxels)")
    
    def gaussian_filter_smooth(self, sigma=1.0):
        """Apply Gaussian filtering for smoothing"""
        self.processed_data = gaussian_filter(
            self.processed_data.astype(np.float64), 
            sigma=sigma
        )
        print(f"Applied Gaussian filter (sigma={sigma})")
    
    def morphological_operations(self, operation='opening', structure_size=3, iterations=1):
        """Apply morphological operations"""
        if len(self.processed_data.shape) == 2:
            structure = morphology.disk(structure_size)
        else:
            structure = morphology.ball(structure_size)
        
        if operation == 'opening':
            self.processed_data = morphology.binary_opening(
                self.processed_data.astype(bool), 
                structure
            ).astype(np.uint8)
        elif operation == 'closing':
            self.processed_data = morphology.binary_closing(
                self.processed_data.astype(bool), 
                structure
            ).astype(np.uint8)
        elif operation == 'erosion':
            self.processed_data = morphology.binary_erosion(
                self.processed_data.astype(bool), 
                structure
            ).astype(np.uint8)
        elif operation == 'dilation':
            self.processed_data = morphology.binary_dilation(
                self.processed_data.astype(bool), 
                structure
            ).astype(np.uint8)
        
        print(f"Applied {operation} (structure_size={structure_size})")
    
    def fill_holes(self):
        """Fill holes in binary mask"""
        if len(self.processed_data.shape) == 2:
            self.processed_data = ndimage.binary_fill_holes(
                self.processed_data.astype(bool)
            ).astype(np.uint8)
        else:
            # For 3D, fill holes slice by slice
            for i in range(self.processed_data.shape[0]):
                self.processed_data[i] = ndimage.binary_fill_holes(
                    self.processed_data[i].astype(bool)
                ).astype(np.uint8)
        print("Filled holes")
    
    def femur_specific_processing(self):
        """Femur-specific processing pipeline"""
        # if config is None:
        #     # Optimized for femur bone
        config = {
            'threshold': 0.3,              # Lower threshold for bone
            'gaussian_sigma': 0.5,         # Smooth bone edges
            'median_size': 2,              # Remove noise
            'opening_size': 1,             # Remove small connections
            'fill_holes': False,            # Fill bone interior
            'min_object_size': 5000,       # Large minimum size for femur
            'keep_largest': True,          # Keep main femur bone
            'clear_border': False,         # Don't clear border (femur may touch)
            'closing_size': 4,             # Smooth bone surface
            'final_gaussian_sigma': 0.1,   # Final smoothing
            'remove_small_holes': False,    # Remove small holes in bone
            'connectivity_3d': True,        # Use 3D connectivity
            
        }
        
        print("Starting femur-specific processing...")
        self.process_pipeline(config)
        
        # Additional femur-specific post-processing
        self._remove_small_holes_3d()
        self._smooth_bone_surface()
        
        print("Femur processing completed!")
    
    def _remove_small_holes_3d(self, max_hole_size=1000):
        """Remove small holes in 3D volume"""
        # Invert to make holes into objects
        inverted = 1 - self.processed_data.astype(bool)
        
        # Label holes
        labeled_holes = measure.label(inverted, connectivity=3)
        
        # Remove small holes
        for region in measure.regionprops(labeled_holes):
            if region.area < max_hole_size:
                coords = region.coords
                self.processed_data[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        
        print(f"Removed holes smaller than {max_hole_size} voxels")
    
    def _smooth_bone_surface(self):
        """Apply surface smoothing specific to bone"""
        # Light gaussian smoothing followed by re-thresholding
        smoothed = gaussian_filter(self.processed_data.astype(np.float64), sigma=0.5)
        self.processed_data = (smoothed > 0.5).astype(np.uint8)
        print("Applied bone surface smoothing")
    
    def clear_border_objects(self):
        """Remove objects touching image border"""
        self.processed_data = clear_border(
            self.processed_data.astype(bool)
        ).astype(np.uint8)
        print("Cleared border objects")
    
    def median_filter(self, size=3):
        """Apply median filter to reduce noise"""
        if len(self.processed_data.shape) == 2:
            self.processed_data = ndimage.median_filter(
                self.processed_data, 
                size=size
            )
        else:
            # 3D median filter
            self.processed_data = ndimage.median_filter(
                self.processed_data, 
                size=[size, size, size]
            )
        print(f"Applied median filter (size={size})")
    
    def process_pipeline(self, config):
        """Run complete processing pipeline"""
        print("Starting processing pipeline...")
        
        # Thresholding (if specified)
        if 'threshold' in config:
            self.threshold_image(config['threshold'])
        
        # Gaussian smoothing (before morphological operations)
        if 'gaussian_sigma' in config:
            self.gaussian_filter_smooth(config['gaussian_sigma'])
            # Re-threshold after gaussian
            if 'threshold' in config:
                self.threshold_image(config['threshold'])
        
        # Median filter
        if 'median_size' in config:
            self.median_filter(config['median_size'])
        
        # Morphological opening (remove small noise)
        if 'opening_size' in config:
            self.morphological_operations('opening', config['opening_size'])
        
        # Fill holes
        if config.get('fill_holes', False):
            self.fill_holes()
        
        # Remove small objects
        if 'min_object_size' in config:
            self.remove_small_objects(config['min_object_size'])
        
        # Keep largest component
        if config.get('keep_largest', False):
            self.keep_largest_component()
        
        # Clear border objects
        if config.get('clear_border', False):
            self.clear_border_objects()
        
        # Morphological closing (smooth boundaries)
        if 'closing_size' in config:
            self.morphological_operations('closing', config['closing_size'])
        
        # Final gaussian smoothing
        if 'final_gaussian_sigma' in config:
            self.gaussian_filter_smooth(config['final_gaussian_sigma'])
        
        print("Processing pipeline completed!")
    
    def remove_small_objects(self, min_size=100):
        """Remove small connected components"""
        self.processed_data = morphology.remove_small_objects(
            self.processed_data.astype(bool), 
            min_size=min_size
        ).astype(np.uint8)
        print(f"Removed objects smaller than {min_size} voxels")
    
    def save_dicom_directory(self, output_dir, rescale_to_original=True):
        """Save processed volume as DICOM files"""
        if self.dicom_files is None or len(self.dicom_files) == 0:
            print("No DICOM files loaded")
            return False
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, input_file in enumerate(self.dicom_files):
                # Load original dataset
                original_ds = pydicom.dcmread(input_file, force=True)
                
                # Create new dataset
                new_dataset = original_ds.copy()
                
                # Get processed slice
                if i < self.processed_data.shape[0]:
                    output_slice = self.processed_data[i].copy()
                    
                    if rescale_to_original:
                        # Rescale to original intensity range
                        original_slice = pydicom.dcmread(input_file, force=True).pixel_array
                        if output_slice.dtype != original_slice.dtype:
                            if original_slice.max() > 1:
                                output_slice = output_slice * original_slice.max()
                            output_slice = output_slice.astype(original_slice.dtype)
                    
                    # Update pixel data
                    new_dataset.PixelData = output_slice.tobytes()
                    
                    # Update series description
                    new_dataset.SeriesDescription = "Femur-processed " + str(getattr(new_dataset, 'SeriesDescription', ''))
                    
                    # Generate output filename
                    input_name = Path(input_file).stem
                    output_file = output_path / f"{input_name}_processed.dcm"
                    
                    # Save
                    new_dataset.save_as(str(output_file))
                    
                    if i % 10 == 0:  # Progress indicator
                        print(f"Processed {i+1}/{len(self.dicom_files)} files")
            
            print(f"Saved all processed files to: {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving DICOM directory: {e}")
            return False
    
    def get_stats(self):
        """Get processing statistics"""
        if self.processed_data is not None:
            return {
                'shape': self.processed_data.shape,
                'dtype': self.processed_data.dtype,
                'min_value': self.processed_data.min(),
                'max_value': self.processed_data.max(),
                'mean_value': self.processed_data.mean(),
                'non_zero_voxels': np.count_nonzero(self.processed_data)
            }
        return None

# Femur-specific processing functions
def process_femur_directory(input_dir, output_dir):
    """Process directory of femur DICOM files"""
    
    processor = FemurDICOMProcessor()
    
    # Load DICOM directory
    if not processor.load_dicom_directory(input_dir):
        return False
    
    
    
    # Run femur-specific processing
    processor.femur_specific_processing()
    
    # Print statistics
    stats = processor.get_stats()
    print(f"Final stats: {stats}")
    
    # Save results
    return processor.save_dicom_directory(output_dir)



if __name__ == "__main__":
    input_directory = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/pred/CT3/SL015_left_CT3_prediction_class1_dcm"
    output_directory = "/home/user/auto-annotation/newWeightauto-annotation/dataset/femur-bone/data/final/CT3/SL015_left_CT3_prediction_class1_dcm"
    
    print("Processing femur DICOM directory...")
    success = process_femur_directory(input_directory, output_directory)
    
    if success:
        print("Femur processing completed successfully!")
    else:
        print("✗ Femur processing failed!")