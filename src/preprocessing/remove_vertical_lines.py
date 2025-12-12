"""
竖线去除工具

用于去除作文纸图片中的竖直格子线，保留横线和手写笔迹。
采用智能策略：只删除长竖线中没有笔迹横过的部分。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image


def detect_vertical_lines(image: np.ndarray, 
                         line_height: int = 100,
                         min_length_factor: float = 2.0,
                         angle_range: Tuple[float, float] = (85, 95)) -> List[Tuple[int, int, int]]:
    """
    检测图片中的长竖线
    
    参数:
        image: 输入图片 (灰度图)
        line_height: 行高（像素），用于计算最小长度阈值
        min_length_factor: 最小长度系数，竖线长度必须 > line_height * min_length_factor
        angle_range: 角度范围 (最小角度, 最大角度)，用于筛选垂直线
    
    返回:
        List of (x, y1, y2): 竖线的x坐标和起止y坐标
    """
    # 边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # 霍夫变换检测直线
    min_length = int(line_height * min_length_factor)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=min_length, maxLineGap=20)
    
    if lines is None:
        return []
    
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算角度（相对于水平线）
        if x2 - x1 == 0:
            angle = 90  # 完美垂直
        else:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        
        # 筛选垂直线
        if angle_range[0] <= angle <= angle_range[1]:
            # 确保y1 < y2
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 计算线段长度
            length = y2 - y1
            
            if length >= min_length:
                # 使用x坐标的平均值（如果线略微倾斜）
                x_avg = (x1 + x2) // 2
                vertical_lines.append((x_avg, y1, y2))
    
    return vertical_lines


def has_ink_crossing(image: np.ndarray, 
                     x: int, 
                     y: int, 
                     window_size: int = 10,
                     ink_threshold: int = 200) -> bool:
    """
    检测竖线位置(x, y)是否有笔迹横过
    
    参数:
        image: 输入图片 (灰度图，已去除竖线的版本)
        x: 竖线的x坐标
        y: 检测的y坐标
        window_size: 左右窗口大小（像素）
        ink_threshold: 笔迹阈值，像素值 < ink_threshold 认为是笔迹
    
    返回:
        True if 左右两侧有笔迹，False otherwise
    """
    height, width = image.shape[:2]
    
    # 检查边界
    if y < 0 or y >= height or x < 0 or x >= width:
        return False
    
    # 左侧窗口 (跳过竖线本身，向左检测)
    left_start = max(0, x - window_size)
    left_window = image[y, left_start:x] if x > 0 else np.array([])
    
    # 右侧窗口 (跳过竖线本身，向右检测)
    right_end = min(width, x + window_size + 1)
    right_window = image[y, x+1:right_end] if x < width - 1 else np.array([])
    
    # 检测黑色像素（笔迹）- 只有两侧都有深色才算交叉
    left_has_ink = np.any(left_window < ink_threshold) if len(left_window) > 0 else False
    right_has_ink = np.any(right_window < ink_threshold) if len(right_window) > 0 else False
    
    # 必须左右两侧都有笔迹才算横过
    return left_has_ink and right_has_ink


def remove_vertical_lines(image: np.ndarray,
                         line_height: int = 100,
                         window_size: int = 10,
                         ink_threshold: int = 200,
                         visualize: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    智能去除竖直格子线
    
    只删除长竖线中没有笔迹横过的部分，保护有笔迹交叉的区域
    
    参数:
        image: 输入图片 (BGR彩色图或灰度图)
        line_height: 行高（像素）
        window_size: 检测笔迹的左右窗口大小
        ink_threshold: 笔迹阈值
        visualize: 是否生成可视化图片
    
    返回:
        processed_image: 处理后的图片
        visualization: 可视化图片 (如果visualize=True)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 检测竖线
    vertical_lines = detect_vertical_lines(gray, line_height)
    
    print(f"检测到 {len(vertical_lines)} 条竖线")
    
    if len(vertical_lines) == 0:
        return result, None
    
    # 创建临时图像，先把所有竖线位置涂白（用于笔迹检测）
    gray_no_vlines = gray.copy()
    for x, y1, y2 in vertical_lines:
        if x < gray.shape[1]:
            gray_no_vlines[y1:y2+1, x] = 255
    
    # 创建掩码
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    vis_image = result.copy() if visualize else None
    
    removed_pixels = 0
    protected_pixels = 0
    
    # 对每条竖线进行处理
    for line_idx, (x, y1, y2) in enumerate(vertical_lines):
        if line_idx % 10 == 0:
            print(f"  处理竖线 {line_idx+1}/{len(vertical_lines)}")
        
        # 逐像素检查
        for y in range(y1, y2 + 1):
            if y >= gray.shape[0]:
                break
            
            # 在去除竖线的图像上检测笔迹
            if has_ink_crossing(gray_no_vlines, x, y, window_size, ink_threshold):
                # 保护这个像素（有笔迹横过）
                protected_pixels += 1
                if visualize and vis_image is not None:
                    # 用绿色标记保护区域
                    vis_image[y, x] = [0, 255, 0]
            else:
                # 删除这个像素（纯净的格子线）
                mask[y, x] = 0
                removed_pixels += 1
                if visualize and vis_image is not None:
                    # 用黄色标记删除区域
                    vis_image[y, x] = [0, 255, 255]
    
    print(f"删除像素数: {removed_pixels}")
    print(f"保护像素数: {protected_pixels}")
    
    # 使用图像修复填充删除的竖线
    if len(image.shape) == 3:
        result = cv2.inpaint(result, 255 - mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        result_bgr = cv2.inpaint(result_bgr, 255 - mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    
    return result, vis_image


def process_image_file(input_path: str,
                      output_path: Optional[str] = None,
                      vis_path: Optional[str] = None,
                      line_height: int = 100,
                      window_size: int = 10,
                      ink_threshold: int = 200) -> bool:
    """
    处理图片文件，去除竖线
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径 (如果为None，自动生成)
        vis_path: 可视化图片路径 (如果为None，不生成)
        line_height: 行高（像素）
        window_size: 检测窗口大小
        ink_threshold: 笔迹阈值
    
    返回:
        True if 成功, False otherwise
    """
    try:
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            print(f"错误: 无法读取图片 {input_path}")
            return False
        
        print(f"处理图片: {input_path}")
        print(f"图片尺寸: {image.shape[1]} x {image.shape[0]}")
        
        # 处理
        result, vis = remove_vertical_lines(
            image,
            line_height=line_height,
            window_size=window_size,
            ink_threshold=ink_threshold,
            visualize=(vis_path is not None)
        )
        
        # 保存结果
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_no_vlines{input_path_obj.suffix}")
        
        cv2.imwrite(output_path, result)
        print(f"[OK] 结果保存到: {output_path}")
        
        # 保存可视化
        if vis_path and vis is not None:
            cv2.imwrite(vis_path, vis)
            print(f"[OK] 可视化保存到: {vis_path}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """
    测试代码
    """
    import argparse
    from pathlib import Path
    
    # 默认测试图片路径
    default_input = Path(__file__).parent.parent.parent / "output" / "temp_cropped.png"
    
    parser = argparse.ArgumentParser(description="去除图片中的竖直格子线")
    parser.add_argument("--input", type=str, default=str(default_input),
                       help="输入图片路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出图片路径 (默认: 输入文件名_no_vlines)")
    parser.add_argument("--vis", type=str, default=None,
                       help="可视化输出路径 (默认: 不生成)")
    parser.add_argument("--line-height", type=int, default=100,
                       help="行高（像素，默认100）")
    parser.add_argument("--window-size", type=int, default=10,
                       help="检测窗口大小（像素，默认10）")
    parser.add_argument("--ink-threshold", type=int, default=200,
                       help="笔迹阈值（0-255，默认200）")
    
    args = parser.parse_args()
    
    print("="*80)
    print("竖线去除工具")
    print("="*80)
    print(f"参数配置:")
    print(f"  行高: {args.line_height}px")
    print(f"  检测窗口: {args.window_size}px")
    print(f"  笔迹阈值: {args.ink_threshold}")
    print("="*80)
    
    # 检查输入文件
    if not Path(args.input).exists():
        print(f"\n错误: 输入文件不存在: {args.input}")
        print(f"请先运行 02_cropped_ocr_test.ipynb 生成裁剪后的测试图片")
        exit(1)
    
    # 设置默认可视化路径
    if args.vis is None and args.output:
        output_path = Path(args.output)
        args.vis = str(output_path.parent / f"{output_path.stem}_vis{output_path.suffix}")
    
    # 处理图片
    success = process_image_file(
        input_path=args.input,
        output_path=args.output,
        vis_path=args.vis,
        line_height=args.line_height,
        window_size=args.window_size,
        ink_threshold=args.ink_threshold
    )
    
    if success:
        print("\n" + "="*80)
        print("[OK] 处理完成!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("[FAIL] 处理失败")
        print("="*80)
        exit(1)
