"""
竖线删除模块

删除图像中检测到的竖线
"""

import cv2
import numpy as np
from typing import List, Tuple, Literal


def remove_vertical_lines(
    image: np.ndarray,
    lines: List[Tuple[int, int, int, float]],
    line_width_half: int = 3,
    method: Literal['paint', 'inpaint'] = 'paint',
    inpaint_radius: int = 3
) -> np.ndarray:
    """
    删除竖线
    
    参数:
        image: 输入图像(灰度图),会创建副本不修改原图
        lines: 竖线列表 [(x, y1, y2, angle), ...]
        line_width_half: 删除宽度的一半(像素)
            - 例如line_width_half=3,删除范围为x±3,共7像素宽
        method: 删除方法
            - 'paint': 直接涂白(快速,适合格子纸)
            - 'inpaint': 使用OpenCV修复算法(慢,效果更自然)
        inpaint_radius: 修复半径(仅method='inpaint'时使用)
    
    返回:
        处理后的图像副本
    
    示例:
        >>> img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
        >>> lines = [(500, 100, 2800, 90), (600, 100, 2800, 90)]
        >>> # 删除7px宽度(x±3)
        >>> result = remove_vertical_lines(img, lines, line_width_half=3)
    """
    result = image.copy()
    
    if method == 'paint':
        # 方法1: 直接涂白
        for x, y1, y2, angle in lines:
            # 计算删除区域
            x_start = max(0, x - line_width_half)
            x_end = min(result.shape[1], x + line_width_half + 1)
            
            # 涂白
            result[y1:y2+1, x_start:x_end] = 255
    
    elif method == 'inpaint':
        # 方法2: 使用OpenCV的inpaint修复
        # 先创建mask
        mask = np.zeros(result.shape, dtype=np.uint8)
        for x, y1, y2, angle in lines:
            x_start = max(0, x - line_width_half)
            x_end = min(mask.shape[1], x + line_width_half + 1)
            mask[y1:y2+1, x_start:x_end] = 255
        
        # 使用Telea算法修复
        result = cv2.inpaint(result, mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'paint' or 'inpaint'")
    
    return result


def remove_vertical_lines_iterative(
    image: np.ndarray,
    detect_func,
    merge_func,
    num_iterations: int = 2,
    removal_params_list: List[dict] = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    迭代删除竖线
    
    Pipeline: 检测 → 合并 → 删除 → 再次检测 → 再次合并 → 再次删除
    
    参数:
        image: 输入图像(灰度图)
        detect_func: 检测函数,签名为 detect_func(image, **kwargs) -> lines
        merge_func: 合并函数,签名为 merge_func(lines, **kwargs) -> merged_lines
        num_iterations: 迭代次数(默认2次)
        removal_params_list: 每次删除的参数列表
            - 例如: [{'line_width_half': 3}, {'line_width_half': 2}]
            - 如果为None,使用默认参数
    
    返回:
        (final_image, iteration_info):
            - final_image: 最终处理后的图像
            - iteration_info: 每次迭代的统计信息列表
    
    示例:
        >>> from preprocessing import detect_vertical_lines, merge_vertical_lines
        >>> img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
        >>> 
        >>> # 定义检测和合并函数
        >>> def detect(img):
        ...     return detect_vertical_lines(img, min_length=200, threshold=100)
        >>> def merge(lines):
        ...     return merge_vertical_lines(lines, x_threshold=3)
        >>> 
        >>> # 迭代删除
        >>> final_img, info = remove_vertical_lines_iterative(
        ...     img, detect, merge, num_iterations=2,
        ...     removal_params_list=[{'line_width_half': 3}, {'line_width_half': 2}]
        ... )
    """
    if removal_params_list is None:
        # 默认参数:第一次删除宽度大,第二次删除宽度小
        removal_params_list = [
            {'line_width_half': 3, 'method': 'paint'},  # 7px
            {'line_width_half': 2, 'method': 'paint'}   # 5px
        ]
    
    # 确保参数列表长度与迭代次数匹配
    if len(removal_params_list) < num_iterations:
        # 复制最后一个参数
        last_params = removal_params_list[-1] if removal_params_list else {'line_width_half': 3}
        removal_params_list.extend([last_params] * (num_iterations - len(removal_params_list)))
    
    current_image = image.copy()
    iteration_info = []
    
    for i in range(num_iterations):
        iteration_data = {'iteration': i + 1}
        
        # 1. 检测竖线
        lines_raw = detect_func(current_image)
        iteration_data['lines_detected'] = len(lines_raw)
        
        # 2. 合并竖线
        lines_merged = merge_func(lines_raw)
        iteration_data['lines_merged'] = len(lines_merged)
        iteration_data['lines_removed_by_merge'] = len(lines_raw) - len(lines_merged)
        
        # 3. 删除竖线
        removal_params = removal_params_list[i]
        current_image = remove_vertical_lines(current_image, lines_merged, **removal_params)
        iteration_data['removal_params'] = removal_params
        
        iteration_info.append(iteration_data)
    
    return current_image, iteration_info


def visualize_removal_comparison(
    original: np.ndarray,
    after_round1: np.ndarray,
    after_round2: np.ndarray,
    round1_info: dict,
    round2_info: dict
) -> str:
    """
    生成两次删除的对比报告
    
    参数:
        original: 原图
        after_round1: 第一轮删除后
        after_round2: 第二轮删除后
        round1_info: 第一轮统计信息
        round2_info: 第二轮统计信息
    
    返回:
        格式化的报告字符串
    """
    report = f"""
{'='*80}
两次迭代删除统计报告
{'='*80}

第一轮删除:
  检测到: {round1_info['lines_detected']} 条竖线
  合并后: {round1_info['lines_merged']} 条竖线
  删除宽度: {round1_info['removal_params']['line_width_half']*2+1}px
  
第二轮删除(清理残留):
  检测到: {round2_info['lines_detected']} 条残留竖线
  合并后: {round2_info['lines_merged']} 条残留竖线
  删除宽度: {round2_info['removal_params']['line_width_half']*2+1}px

总计删除:
  第一轮: {round1_info['lines_merged']} 条
  第二轮: {round2_info['lines_merged']} 条
  累计: {round1_info['lines_merged'] + round2_info['lines_merged']} 条

{'='*80}
"""
    return report
