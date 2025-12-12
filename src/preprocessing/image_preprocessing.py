"""
图像预处理模块
提供多种预处理方法用于OCR前的图像增强
"""
import cv2
import numpy as np


def baseline_preprocess(cell_img):
    """
    方法1: Baseline（基准方法）
    作用: 仅将彩色图像转换为灰度图
    
    参数:
        cell_img: OpenCV图像 (BGR或灰度)
        
    返回:
        gray_bgr: BGR格式的灰度图像 (H, W, 3), dtype=uint8
    """
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
    
    # 转换为3通道BGR格式（PaddleOCR要求）
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_bgr


def adaptive_threshold_preprocess(cell_img, block_size=11, C=2):
    """
    方法2: Adaptive Threshold（自适应二值化）
    作用: 根据局部区域自适应计算阈值，将图像二值化
    
    参数:
        cell_img: OpenCV图像 (BGR或灰度)
        block_size: 邻域窗口大小（奇数），默认11
        C: 阈值调整常数，默认2
        
    返回:
        binary_bgr: BGR格式的二值图像 (H, W, 3), dtype=uint8, 值为0或255
    """
    # 先转灰度
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
    
    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )
    
    # 转换为3通道BGR格式（PaddleOCR要求）
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary_bgr


def resize_normalize_preprocess(cell_img, target_height=48, interpolation=cv2.INTER_CUBIC):
    """
    方法3: Resize Normalize（尺寸标准化）
    作用: 将格子图像缩放到统一高度，保持宽高比，不足的部分用白色填充
    
    参数:
        cell_img: OpenCV图像 (BGR或灰度)
        target_height: 目标高度，默认48像素
        interpolation: 插值方法，默认cv2.INTER_CUBIC（三次插值）
        
    返回:
        normalized_bgr: BGR格式的标准化尺寸图像 (target_height, new_width, 3), dtype=uint8
    """
    # 先转灰度
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
    
    h, w = gray.shape
    
    # 如果高度已经是目标高度，转换后返回
    if h == target_height:
        normalized_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return normalized_bgr
    
    # 计算缩放比例（保持宽高比）
    scale = target_height / h
    new_width = int(w * scale)
    
    # 缩放图像
    resized = cv2.resize(gray, (new_width, target_height), interpolation=interpolation)
    
    # 如果宽度不足，进行白色填充（右侧填充）
    if new_width < w:
        # 需要填充
        normalized = cv2.copyMakeBorder(
            resized,
            0, 0, 0, w - new_width,
            cv2.BORDER_CONSTANT,
            value=255  # 白色填充
        )
    else:
        normalized = resized
    
    # 转换为3通道BGR格式（PaddleOCR要求）
    normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized_bgr


def enhancement_preprocess(cell_img, open_kernel=(2, 2), close_kernel=(3, 3), sharpen_alpha=0.5):
    """
    方法4: Enhancement（字符增强）
    作用: 三步增强字符质量
        1. 开运算: 去除小噪点
        2. 闭运算: 填补断笔、连接笔画
        3. 锐化: 增强边缘对比度
    
    参数:
        cell_img: OpenCV图像 (BGR或灰度)
        open_kernel: 开运算核大小，默认(2, 2)
        close_kernel: 闭运算核大小，默认(3, 3)
        sharpen_alpha: 锐化强度，默认0.5
        
    返回:
        enhanced_bgr: BGR格式的增强后图像 (H, W, 3), dtype=uint8
    """
    # 先转灰度
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
    
    # 步骤1: 开运算（去噪）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open)
    
    # 步骤2: 闭运算（连接笔画）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 步骤3: 锐化（增强边缘）
    # 使用拉普拉斯算子
    laplacian = cv2.Laplacian(closed, cv2.CV_64F)
    sharpened = closed - sharpen_alpha * laplacian
    
    # 转换回uint8并限制范围
    enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # 转换为3通道BGR格式（PaddleOCR要求）
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr


# 预处理方法字典（方便循环调用）
PREPROCESSING_METHODS = {
    'baseline': baseline_preprocess,
    'adaptive_threshold': adaptive_threshold_preprocess,
    'resize_normalize': resize_normalize_preprocess,
    'enhancement': enhancement_preprocess
}


def get_preprocessing_method(method_name):
    """
    根据名称获取预处理函数
    
    参数:
        method_name: 方法名称，可选值: 'baseline', 'adaptive_threshold', 'resize_normalize', 'enhancement'
        
    返回:
        preprocess_fn: 预处理函数
    """
    if method_name not in PREPROCESSING_METHODS:
        raise ValueError(f"未知的预处理方法: {method_name}，可选值: {list(PREPROCESSING_METHODS.keys())}")
    
    return PREPROCESSING_METHODS[method_name]
