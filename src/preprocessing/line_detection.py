"""
竖线检测模块

使用霍夫变换检测图像中的竖线
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def detect_vertical_lines(
    image: np.ndarray,
    min_length: int = 200,
    threshold: int = 100,
    angle_range: Tuple[float, float] = (85, 95),
    max_gap: int = 20,
    canny_low: int = 50,
    canny_high: int = 150
) -> List[Tuple[int, int, int, float]]:
    """
    使用霍夫变换检测竖线
    
    参数:
        image: 输入图像(灰度图或BGR图)
        min_length: 最小线段长度阈值(像素)
        threshold: 霍夫变换累加器阈值(越高越严格)
        angle_range: 角度范围(min_angle, max_angle),单位:度
        max_gap: 允许的最大间隙(像素)
        canny_low: Canny边缘检测低阈值
        canny_high: Canny边缘检测高阈值
    
    返回:
        List[(x, y1, y2, angle)]: 检测到的竖线列表
            x: 竖线的x坐标(平均值)
            y1: 起点y坐标(较小值)
            y2: 终点y坐标(较大值)
            angle: 竖线角度(度)
    
    示例:
        >>> import cv2
        >>> img = cv2.imread('test.png')
        >>> lines = detect_vertical_lines(img, min_length=200, threshold=100)
        >>> print(f"检测到 {len(lines)} 条竖线")
    """
    # 1. 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2. Canny边缘检测
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    
    # 3. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=threshold,
        minLineLength=min_length, 
        maxLineGap=max_gap
    )
    
    if lines is None:
        return []
    
    # 4. 筛选竖线
    angle_min, angle_max = angle_range
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算角度
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        
        # 筛选角度在范围内的竖线
        if angle_min <= angle <= angle_max:
            # 确保y1 < y2
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 计算长度
            length = y2 - y1
            
            # 再次检查长度(因为可能是斜线)
            if length >= min_length:
                x_avg = int((x1 + x2) / 2)
                vertical_lines.append((x_avg, y1, y2, angle))
    
    return vertical_lines


def visualize_vertical_lines(
    image: np.ndarray,
    lines: List[Tuple[int, int, int, float]],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3
) -> np.ndarray:
    """
    在图像上可视化竖线
    
    参数:
        image: 输入图像(BGR或RGB)
        lines: 竖线列表 [(x, y1, y2, angle), ...]
        color: 线条颜色(B, G, R)或(R, G, B)
        thickness: 线条粗细
    
    返回:
        绘制了竖线的图像副本
    """
    vis_img = image.copy()
    
    for x, y1, y2, angle in lines:
        cv2.line(vis_img, (x, y1), (x, y2), color, thickness)
    
    return vis_img


def get_vertical_lines_statistics(lines: List[Tuple[int, int, int, float]]) -> dict:
    """
    统计竖线的基本信息
    
    参数:
        lines: 竖线列表 [(x, y1, y2, angle), ...]
    
    返回:
        统计信息字典
    """
    if not lines:
        return {
            'count': 0,
            'x_range': (0, 0),
            'y_range': (0, 0),
            'avg_length': 0,
            'angle_range': (0, 0)
        }
    
    x_coords = [line[0] for line in lines]
    y1_coords = [line[1] for line in lines]
    y2_coords = [line[2] for line in lines]
    lengths = [line[2] - line[1] for line in lines]
    angles = [line[3] for line in lines]
    
    return {
        'count': len(lines),
        'x_range': (min(x_coords), max(x_coords)),
        'y_range': (min(y1_coords), max(y2_coords)),
        'avg_length': np.mean(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'angle_range': (min(angles), max(angles)),
        'avg_angle': np.mean(angles)
    }
