"""
Pipeline模块 - 端到端OCR识别管线
"""
from .image_cropper import crop_image, auto_crop, CROP_REGIONS
# from .end_to_end_ocr import EndToEndOCRPipeline

__all__ = [
    'crop_image',
    'auto_crop',
    'CROP_REGIONS',
    'EndToEndOCRPipeline'
]
