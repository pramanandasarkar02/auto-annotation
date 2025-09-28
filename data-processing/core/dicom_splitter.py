import logging
import numpy as np
import SimpleITK as sitk

class DicomSplitter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def split_left_right(self, image):
        """Split image into left and right halves"""
        self.logger.info("Splitting DICOM image into left and right halves")
        array = sitk.GetArrayFromImage(image)
        self.logger.info(f"Original image shape: {array.shape}")
        
        mid_x = array.shape[2] // 2
        left_array = array[:, :, :mid_x]
        right_array = array[:, :, mid_x:]
        
        self.logger.info(f"Left shape: {left_array.shape}, Right shape: {right_array.shape}")

        left_img = sitk.GetImageFromArray(left_array)
        right_img = sitk.GetImageFromArray(right_array)

        spacing = image.GetSpacing()
        origin = list(image.GetOrigin())
        direction = image.GetDirection()

        right_origin = origin[:]
        right_origin[0] += mid_x * spacing[0]

        for img, org in zip([left_img, right_img], [origin, right_origin]):
            img.SetSpacing(spacing)
            img.SetOrigin(org)
            img.SetDirection(direction)

        return left_img, right_img