
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import logging


def save_image_and_label_to_train_dir(image, image_path, label, label_path):
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, affine=np.eye(4)), label_path)


# def match_label_to_image(image, label):
#     print(image.shape, label.shape)
#     for perm in [(0,1,2), (2,1,0), (0,2,1), (1,0,2), (1,2,0), (2,0,1)]:
#         transposed = np.transpose(label, perm)
#         if transposed.shape == image.shape:
#             return transposed
#     scale_factors = [i / l for i, l in zip(image.shape, label.shape)]
#     print(f"[!] Resizing label from {label.shape} to {image.shape} using scale {scale_factors}")
#     resized_label = zoom(label, scale_factors, order=0)  # nearest neighbor for segmentation
#     return resized_label

def match_label_to_image(image, label):
    print(f"Image shape: {image.shape}, Label shape: {label.shape}")
    
    # Dynamic squeezing: remove dimensions that don't contribute meaningful information
    def smart_squeeze(arr, target_shape):
        current_shape = list(arr.shape)
        target_dims = len(target_shape)
        
        # First, squeeze all singleton dimensions (size 1)
        arr = np.squeeze(arr)
        print(f"After squeezing singletons: {arr.shape}")
        
        # If we still have more dimensions than target, handle strategically
        while len(arr.shape) > target_dims:
            current_shape = list(arr.shape)
            
            # Find the smallest dimension that's not 1
            non_singleton_dims = [(i, size) for i, size in enumerate(current_shape) if size > 1]
            
            if len(non_singleton_dims) <= target_dims:
                # We have the right number of meaningful dimensions, break
                break
            
            # Find dimensions that might be redundant
            if len(current_shape) == 4 and target_dims == 3:
                # Common case: 4D to 3D
                # Look for dimensions that might represent channels/time that we can reduce
                candidates = []
                for i, size in enumerate(current_shape):
                    if size <= 3:  # Likely a channel dimension
                        candidates.append((i, size))
                
                if candidates:
                    # Take the first channel of the smallest channel dimension
                    dim_to_reduce = min(candidates, key=lambda x: x[1])[0]
                    print(f"Reducing dimension {dim_to_reduce} (size {current_shape[dim_to_reduce]}) by taking first slice")
                    arr = np.take(arr, 0, axis=dim_to_reduce)
                else:
                    # No clear channel dimension, take first slice of largest dimension
                    largest_dim = max(range(len(current_shape)), key=lambda i: current_shape[i])
                    print(f"No clear channel dim, reducing largest dimension {largest_dim}")
                    arr = np.take(arr, 0, axis=largest_dim)
            else:
                # General case: take first slice of the first dimension
                print(f"General reduction: taking first slice of dimension 0")
                arr = arr[0]
            
            print(f"After reduction: {arr.shape}")
        
        return arr
    
    # Handle case where label has fewer dimensions than image
    if len(label.shape) < len(image.shape):
        # Add singleton dimensions to match image dimensions
        while len(label.shape) < len(image.shape):
            label = np.expand_dims(label, axis=-1)
        print(f"Expanded label shape to: {label.shape}")
    
    # Handle case where label has more dimensions than image
    elif len(label.shape) > len(image.shape):
        label = smart_squeeze(label, image.shape)
        print(f"Smart squeezed label shape to: {label.shape}")
    
    # Now try different permutations only if dimensions match
    if len(label.shape) == len(image.shape):
        # Generate all possible permutations for the given number of dimensions
        from itertools import permutations as iter_permutations
        
        dim_count = len(image.shape)
        all_permutations = list(iter_permutations(range(dim_count)))
        
        print(f"Trying {len(all_permutations)} permutations for {dim_count}D arrays")
        
        for perm in all_permutations:
            try:
                transposed = np.transpose(label, perm)
                if transposed.shape == image.shape:
                    print(f"Found matching permutation: {perm}")
                    return transposed
            except ValueError as e:
                print(f"Permutation {perm} failed: {e}")
                continue
    
    # If no permutation works, try to resize the label to match image shape
    if len(label.shape) == len(image.shape):
        print(f"No permutation worked. Resizing label from {label.shape} to {image.shape}")
        scale_factors = [i / l for i, l in zip(image.shape, label.shape)]
        print(f"Scale factors: {scale_factors}")
        
        try:
            resized_label = zoom(label, scale_factors, order=0)  # nearest neighbor for segmentation
            return resized_label
        except Exception as e:
            print(f"Resizing failed: {e}")
    
    # As a last resort, try to reshape if the total number of elements is the same
    if np.prod(label.shape) == np.prod(image.shape):
        print("Attempting reshape as last resort")
        return label.reshape(image.shape)
    else:
        raise ValueError(f"Cannot match label shape {label.shape} to image shape {image.shape}")

def load_images_labels(image_path, label_path):
    image = nib.load(image_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    # Try to match label shape with image shape
    label = match_label_to_image(image, label)
    return image, label


def image_label_shape_alignment(images_dir, labels_dir):
    for series_id in os.listdir(images_dir):
        image_series_path = os.path.join(images_dir, series_id)
        label_series_path = os.path.join(labels_dir, series_id)

        if not os.path.isdir(image_series_path) or not os.path.isdir(label_series_path):
            continue

        for image_id in os.listdir(image_series_path):
            image_path = os.path.join(image_series_path, image_id)

            # Get patient prefix without extension
            base_name = os.path.splitext(os.path.splitext(image_id)[0])[0]

            # Try to find matching label file in label_series_path
            matches = [f for f in os.listdir(label_series_path) if base_name in f]

            if not matches:
                logging.warning(f"No label found for {image_id}")
                continue

            label_path = os.path.join(label_series_path, matches[0])
            
            image, label = load_images_labels(image_path, label_path)
            logging.info(f"Found pair: {os.path.basename(image_path)}({image.shape}) <-> {os.path.basename(label_path)}({label.shape})")

            save_image_and_label_to_train_dir(image, image_path, label, label_path)

    
