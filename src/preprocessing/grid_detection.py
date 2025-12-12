"""
格子检测模块

检测格子纸的横线和竖线，生成格子网格坐标
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def detect_grid_lines(
    image: np.ndarray,
    detect_horizontal: bool = True,
    detect_vertical: bool = True,
    min_length: int = 200,
    threshold: int = 100,
    max_gap: int = 20,
    angle_range_horizontal: Tuple[int, int] = (0, 5),
    angle_range_vertical: Tuple[int, int] = (85, 95),
    canny_low: int = 50,
    canny_high: int = 150
) -> Dict[str, List[Tuple]]:
    """
    检测图像中的横线和竖线
    
    参数:
        image: 输入图像 (灰度图)
        detect_horizontal: 是否检测横线
        detect_vertical: 是否检测竖线
        min_length: 最小线条长度 (像素)
        threshold: 霍夫变换累加器阈值
        max_gap: 允许的最大间隙 (像素)
        angle_range_horizontal: 横线角度范围 (度)
        angle_range_vertical: 竖线角度范围 (度)
        canny_low: Canny边缘检测低阈值
        canny_high: Canny边缘检测高阈值
    
    返回:
        字典包含:
            'horizontal': [(x1, y1, x2, y2, angle), ...] 横线列表
            'vertical': [(x, y1, y2, angle), ...] 竖线列表 (x为平均x坐标)
    """
    # 边缘检测
    edges = cv2.Canny(image, canny_low, canny_high, apertureSize=3)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi/180, 
        threshold=threshold,
        minLineLength=min_length, 
        maxLineGap=max_gap
    )
    
    result = {
        'horizontal': [],
        'vertical': []
    }
    
    if lines is None:
        return result
    
    # 分类横线和竖线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算角度
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        
        # 横线检测
        if detect_horizontal:
            if angle_range_horizontal[0] <= angle <= angle_range_horizontal[1]:
                # 确保x1 < x2
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                result['horizontal'].append((x1, y1, x2, y2, angle))
        
        # 竖线检测
        if detect_vertical:
            if angle_range_vertical[0] <= angle <= angle_range_vertical[1]:
                # 确保y1 < y2
                if y1 > y2:
                    y1, y2 = y2, y1
                x_avg = (x1 + x2) // 2
                result['vertical'].append((x_avg, y1, y2, angle))
    
    return result


def cluster_lines_by_position(
    lines: List[Tuple],
    line_type: str = 'vertical',
    position_threshold: int = 10
) -> List[List[Tuple]]:
    """
    按位置聚类线条 (用于合并接近的线条)
    
    参数:
        lines: 线条列表
        line_type: 'vertical' 或 'horizontal'
        position_threshold: 位置差异阈值 (像素)
    
    返回:
        聚类后的线条组列表
    """
    if not lines:
        return []
    
    # 按位置排序
    if line_type == 'vertical':
        # 竖线按x坐标排序
        sorted_lines = sorted(lines, key=lambda l: l[0])
        get_position = lambda l: l[0]
    else:
        # 横线按y坐标排序
        sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
        get_position = lambda l: (l[1] + l[3]) / 2
    
    # 聚类
    clusters = []
    current_cluster = [sorted_lines[0]]
    
    for i in range(1, len(sorted_lines)):
        current_pos = get_position(sorted_lines[i])
        prev_pos = get_position(current_cluster[0])
        
        if abs(current_pos - prev_pos) <= position_threshold:
            current_cluster.append(sorted_lines[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_lines[i]]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def merge_line_cluster(
    cluster: List[Tuple],
    line_type: str = 'vertical'
) -> Tuple:
    """
    合并一组线条
    
    参数:
        cluster: 线条组
        line_type: 'vertical' 或 'horizontal'
    
    返回:
        合并后的线条
    """
    if line_type == 'vertical':
        # 竖线: (x, y1, y2, angle)
        x_avg = int(np.mean([line[0] for line in cluster]))
        y1_min = min([line[1] for line in cluster])
        y2_max = max([line[2] for line in cluster])
        angle_avg = np.mean([line[3] for line in cluster])
        return (x_avg, y1_min, y2_max, angle_avg)
    else:
        # 横线: (x1, y1, x2, y2, angle)
        x1_min = min([line[0] for line in cluster])
        x2_max = max([line[2] for line in cluster])
        y_avg = int(np.mean([(line[1] + line[3]) / 2 for line in cluster]))
        angle_avg = np.mean([line[4] for line in cluster])
        return (x1_min, y_avg, x2_max, y_avg, angle_avg)


def generate_grid_cells(
    horizontal_lines: List[Tuple],
    vertical_lines: List[Tuple],
    image_shape: Tuple[int, int],
    merge_threshold: int = 20
) -> List[Dict]:
    """
    根据横线和竖线生成格子网格
    
    参数:
        horizontal_lines: 横线列表 [(x1, y1, x2, y2, angle), ...]
        vertical_lines: 竖线列表 [(x, y1, y2, angle), ...]
        image_shape: 图像尺寸 (height, width)
        merge_threshold: 线条合并阈值
    
    返回:
        格子列表，每个格子包含:
            {
                'row': 行索引,
                'col': 列索引,
                'x1': 左边界,
                'y1': 上边界,
                'x2': 右边界,
                'y2': 下边界
            }
    """
    # 合并接近的横线
    h_clusters = cluster_lines_by_position(horizontal_lines, 'horizontal', merge_threshold)
    h_merged = [merge_line_cluster(cluster, 'horizontal') for cluster in h_clusters]
    
    # 过滤短横线：长度小于最大长度一半的认为是错误检测
    if h_merged:
        h_lengths = [line[2] - line[0] for line in h_merged]  # x2 - x1
        max_h_length = max(h_lengths)
        h_merged = [line for line, length in zip(h_merged, h_lengths) 
                    if length >= max_h_length / 2]
    
    h_positions = sorted([line[1] for line in h_merged])  # y坐标
    
    # 合并接近的竖线
    v_clusters = cluster_lines_by_position(vertical_lines, 'vertical', merge_threshold)
    v_merged = [merge_line_cluster(cluster, 'vertical') for cluster in v_clusters]
    
    # 过滤短竖线：长度小于最大长度一半的认为是错误检测
    if v_merged:
        v_lengths = [line[2] - line[1] for line in v_merged]  # y2 - y1
        max_v_length = max(v_lengths)
        v_merged = [line for line, length in zip(v_merged, v_lengths) 
                    if length >= max_v_length / 2]
    
    v_positions = sorted([line[0] for line in v_merged])  # x坐标
    
    # 生成格子
    cells = []
    
    for row_idx in range(len(h_positions) - 1):
        for col_idx in range(len(v_positions) - 1):
            cell = {
                'row': row_idx,
                'col': col_idx,
                'x1': v_positions[col_idx],
                'y1': h_positions[row_idx],
                'x2': v_positions[col_idx + 1],
                'y2': h_positions[row_idx + 1]
            }
            cells.append(cell)
    
    return cells


def is_cell_empty(
    cell_img: np.ndarray,
    threshold: float = 0.95
) -> bool:
    """
    判断格子是否为空
    
    参数:
        cell_img: 格子图像 (BGR 或灰度图)
        threshold: 空白像素比例阈值 (0-1)
    
    返回:
        True if 格子为空
    """
    if cell_img.size == 0:
        return True
    
    # 转换为灰度图（如果需要）
    if len(cell_img.shape) == 3:
        cell_img_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        cell_img_gray = cell_img
    
    # 计算白色像素比例
    white_pixels = np.sum(cell_img_gray > 200)
    total_pixels = cell_img_gray.size
    white_ratio = white_pixels / total_pixels
    
    return white_ratio >= threshold


def visualize_grid(
    image: np.ndarray,
    cells: List[Dict],
    show_empty: bool = False,
    empty_color: Tuple[int, int, int] = (200, 200, 200),
    filled_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    可视化格子网格
    
    参数:
        image: 输入图像
        cells: 格子列表
        show_empty: 是否标记空格子
        empty_color: 空格子颜色 (BGR)
        filled_color: 非空格子颜色 (BGR)
    
    返回:
        可视化图像
    """
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    for cell in cells:
        # 裁剪格子区域
        cell_img = image[cell['y1']:cell['y2'], cell['x1']:cell['x2']]
        
        # 判断是否为空
        is_empty = is_cell_empty(cell_img)
        color = empty_color if is_empty else filled_color
        
        if not is_empty or show_empty:
            # 画格子边框
            cv2.rectangle(
                vis_img,
                (cell['x1'], cell['y1']),
                (cell['x2'], cell['y2']),
                color,
                2
            )
    
    return vis_img


def get_grid_statistics(
    cells: List[Dict],
    image: np.ndarray
) -> Dict:
    """
    统计格子信息
    
    参数:
        cells: 格子列表
        image: 输入图像 (灰度图)
    
    返回:
        统计信息字典
    """
    if not cells:
        return {
            'total_cells': 0,
            'empty_cells': 0,
            'filled_cells': 0,
            'num_rows': 0,
            'num_cols': 0,
            'avg_cell_width': 0,
            'avg_cell_height': 0
        }
    
    # 统计空格子
    empty_count = 0
    for cell in cells:
        cell_img = image[cell['y1']:cell['y2'], cell['x1']:cell['x2']]
        if is_cell_empty(cell_img):
            empty_count += 1
    
    # 行列数
    num_rows = max(cell['row'] for cell in cells) + 1
    num_cols = max(cell['col'] for cell in cells) + 1
    
    # 平均格子尺寸
    widths = [cell['x2'] - cell['x1'] for cell in cells]
    heights = [cell['y2'] - cell['y1'] for cell in cells]
    
    return {
        'total_cells': len(cells),
        'empty_cells': empty_count,
        'filled_cells': len(cells) - empty_count,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'avg_cell_width': np.mean(widths),
        'avg_cell_height': np.mean(heights),
        'cell_width_std': np.std(widths),
        'cell_height_std': np.std(heights)
    }
