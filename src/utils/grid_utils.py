"""
网格处理工具函数

提供格子数据恢复、位置映射等通用功能
"""

from typing import List, Dict, Tuple


def restore_empty_cells(results_list: List[Dict], cells: List[Dict], non_empty_cells: List[Dict]) -> List[Dict]:
    """
    恢复空格子，构建完整的预测结果
    
    参数:
        results_list: OCR识别结果列表（非空格子）
        cells: 所有格子列表（包括空格子）
        non_empty_cells: 非空格子列表
        
    返回:
        prediction_data_full: 完整预测结果（包含空格子）
    """
    # 1. 创建完整的格子字典（包括空格子）
    all_cells_dict = {}
    for cell in cells:
        row, col = cell['row'], cell['col']
        all_cells_dict[(row, col)] = {
            'row': row,
            'col': col,
            'text': '',           # 默认为空
            'confidence': 1.0,    # 空格子置信度设为 1.0
            'is_empty': True      # 标记为空格子
        }
    
    # 2. 填充非空格子的识别结果
    for i, (result, cell) in enumerate(zip(results_list, non_empty_cells)):
        row, col = cell['row'], cell['col']
        all_cells_dict[(row, col)] = {
            'row': row,
            'col': col,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'is_empty': False,
            'ocr_result': result  # 保存完整的 OCR 结果（包含 top_k 等）
        }
    
    # 3. 转换为列表（按 row, col 排序）
    prediction_data_full = sorted(all_cells_dict.values(), 
                                  key=lambda x: (x['row'], x['col']))
    
    return prediction_data_full


def create_cell_position_dict(prediction_data_full: List[Dict]) -> Dict[Tuple[int, int], int]:
    """
    创建 (row, col) → index 映射字典，用于可视化等场景
    
    参数:
        prediction_data_full: 完整的格子预测结果列表
        
    返回:
        cell_dict: {(row, col): index} 映射字典
    """
    return {(cell['row'], cell['col']): i 
            for i, cell in enumerate(prediction_data_full)}
