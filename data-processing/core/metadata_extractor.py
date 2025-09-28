import logging
import SimpleITK as sitk

class MetadataExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_metadata(self, image):
        """Extract comprehensive metadata from image"""
        self.logger.info("Extracting metadata from image")
        return {
            "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
            "size": list(image.GetSize()),
            "depth": image.GetDepth(),
            "pixel_type": str(image.GetPixelIDTypeAsString()),
            "number_of_components": image.GetNumberOfComponentsPerPixel()
        }