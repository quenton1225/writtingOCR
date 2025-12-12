"""
自定义 OCR 后处理框架

提供灵活的 OCR 识别和后处理功能，支持获取概率分布和模块化处理。
"""

from .recognizer import CustomTextRecognizer
from .pipeline import PostProcessingPipeline

__all__ = [
    'CustomTextRecognizer',
    'PostProcessingPipeline',
]

__version__ = '0.1.0'
