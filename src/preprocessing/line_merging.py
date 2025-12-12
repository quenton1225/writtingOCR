"""
竖线合并模块

合并x坐标接近的竖线,处理重复检测和打断的竖线段
"""

import numpy as np
from typing import List, Tuple, Optional


def merge_vertical_lines(
    lines: List[Tuple[int, int, int, float]],
    x_threshold: int = 3,
    y_gap_threshold: Optional[int] = None,
    extend_to_full_range: bool = False
) -> List[Tuple[int, int, int, float]]:
    """
    合并x坐标接近的竖线
    
    处理两类问题:
    1. 同一条竖线被检测成多条(x坐标接近,如x=500,502,504)
    2. 被横线/文字打断的竖线段(x坐标相同,y坐标不连续)
    
    参数:
        lines: 竖线列表 [(x, y1, y2, angle), ...]
        x_threshold: x坐标差距阈值(像素),差距≤此值的竖线会被合并
        y_gap_threshold: y间隙阈值(可选),如果指定则只合并y接近的竖线
        extend_to_full_range: 是否将所有竖线的y范围统一延伸到全局最小y和最大y
                             (默认False,谨慎使用,可能误删文字区域)
    
    返回:
        合并后的竖线列表 [(x_avg, y1_min, y2_max, angle_avg), ...]
            x_avg: 合并后的x坐标(平均值)
            y1_min: 合并后的起点(最小y值)
            y2_max: 合并后的终点(最大y值)
            angle_avg: 合并后的角度(平均值)
    
    示例:
        >>> lines = [(500, 100, 300, 90), (502, 150, 350, 89), (504, 200, 400, 91)]
        >>> merged = merge_vertical_lines(lines, x_threshold=3)
        >>> print(merged)  # [(502, 100, 400, 90)]
    """
    if not lines:
        return []
    
    # 如果需要延伸到全局范围,先计算全局最小和最大y
    if extend_to_full_range:
        global_y_min = min(line[1] for line in lines)
        global_y_max = max(line[2] for line in lines)
    
    # 1. 按x坐标排序
    sorted_lines = sorted(lines, key=lambda line: line[0])
    
    merged = []
    current_group = [sorted_lines[0]]
    
    # 2. 遍历,合并x差距≤threshold的竖线
    for i in range(1, len(sorted_lines)):
        x_curr, y1_curr, y2_curr, angle_curr = sorted_lines[i]
        x_prev = current_group[0][0]
        
        # 检查x坐标是否接近
        if abs(x_curr - x_prev) <= x_threshold:
            # x坐标接近
            
            # 如果指定了y_gap_threshold,额外检查y是否接近
            if y_gap_threshold is not None:
                # 检查与当前组中任意竖线的y是否有重叠或接近
                y_overlap = False
                for _, prev_y1, prev_y2, _ in current_group:
                    # y有重叠或间隙小于阈值
                    if not (y2_curr < prev_y1 - y_gap_threshold or 
                           y1_curr > prev_y2 + y_gap_threshold):
                        y_overlap = True
                        break
                
                if y_overlap:
                    current_group.append(sorted_lines[i])
                else:
                    # x接近但y相距太远,不合并
                    merged_line = _merge_group(current_group)
                    merged.append(merged_line)
                    current_group = [sorted_lines[i]]
            else:
                # 不检查y,直接合并
                current_group.append(sorted_lines[i])
        else:
            # x坐标相差太大,新的一组
            merged_line = _merge_group(current_group)
            merged.append(merged_line)
            current_group = [sorted_lines[i]]
    
    # 3. 处理最后一组
    if current_group:
        merged_line = _merge_group(current_group)
        merged.append(merged_line)
    
    # 如果需要延伸到全局范围,将所有合并后的竖线y范围统一
    if extend_to_full_range:
        merged = [(x, global_y_min, global_y_max, angle) 
                  for x, _, _, angle in merged]
    
    return merged


def _merge_group(group: List[Tuple[int, int, int, float]]) -> Tuple[int, int, int, float]:
    """
    合并一组竖线
    
    参数:
        group: 需要合并的竖线组 [(x, y1, y2, angle), ...]
    
    返回:
        合并后的竖线 (x_avg, y1_min, y2_max, angle_avg)
    """
    x_avg = int(np.mean([line[0] for line in group]))
    y1_min = min([line[1] for line in group])
    y2_max = max([line[2] for line in group])
    angle_avg = np.mean([line[3] for line in group])
    
    return (x_avg, y1_min, y2_max, angle_avg)


def get_merge_statistics(
    original_lines: List[Tuple[int, int, int, float]],
    merged_lines: List[Tuple[int, int, int, float]],
    x_threshold: int = 3
) -> dict:
    """
    统计合并效果
    
    参数:
        original_lines: 原始竖线列表
        merged_lines: 合并后的竖线列表
        x_threshold: x坐标差距阈值
    
    返回:
        统计信息字典
    """
    # 统计每个合并后的竖线包含了多少条原始竖线
    merge_counts = {}
    for merged_line in merged_lines:
        x_merged = merged_line[0]
        count = sum(1 for line in original_lines 
                   if abs(line[0] - x_merged) <= x_threshold)
        if count > 1:
            merge_counts[x_merged] = count
    
    total_merged = len(original_lines) - len(merged_lines)
    merge_locations = len(merge_counts)
    
    return {
        'original_count': len(original_lines),
        'merged_count': len(merged_lines),
        'total_merged': total_merged,
        'merge_locations': merge_locations,
        'merge_details': merge_counts,
        'merge_ratio': total_merged / len(original_lines) if original_lines else 0
    }


def visualize_merge_comparison(
    original_lines: List[Tuple[int, int, int, float]],
    merged_lines: List[Tuple[int, int, int, float]],
    x_threshold: int = 3
) -> str:
    """
    生成合并对比的文本报告
    
    参数:
        original_lines: 原始竖线列表
        merged_lines: 合并后的竖线列表
        x_threshold: x坐标差距阈值
    
    返回:
        格式化的统计报告字符串
    """
    stats = get_merge_statistics(original_lines, merged_lines, x_threshold)
    
    report = f"""
{'='*80}
竖线合并统计报告
{'='*80}

合并前: {stats['original_count']} 条竖线
合并后: {stats['merged_count']} 条竖线
合并了: {stats['total_merged']} 条竖线 ({stats['merge_ratio']*100:.1f}%)
合并位置: {stats['merge_locations']} 个

"""
    
    if stats['merge_details']:
        report += "合并详情(前10个位置):\n"
        for i, (x, count) in enumerate(list(stats['merge_details'].items())[:10]):
            report += f"  x≈{x}: {count}条竖线合并为1条\n"
        
        if stats['merge_locations'] > 10:
            report += f"  ... 还有 {stats['merge_locations']-10} 个合并位置\n"
    else:
        report += "没有发生合并(所有竖线x坐标差距>{x_threshold}px)\n"
    
    report += f"\n{'='*80}\n"
    
    return report
