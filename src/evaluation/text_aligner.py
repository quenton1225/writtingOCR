"""
文本对齐器

处理预测结果和 Ground Truth 的对齐逻辑
"""

import re
from typing import List, Dict, Tuple


class TextAligner:
    """文本对齐器，负责将预测结果和 Ground Truth 按行对齐"""
    
    def __init__(self, empty_char: str = ''):
        """
        初始化对齐器
        
        参数:
            empty_char: 空格子的表示字符
        """
        self.empty_char = empty_char
    
    def prepare_ground_truth(self, ground_truth: str) -> List[List[str]]:
        """
        预处理 Ground Truth，保留行首空格
        
        参数:
            ground_truth: 原始 Ground Truth 字符串（包含换行符和标记）
        
        返回:
            按行分割的字符列表，行首空格用空字符串 '' 表示
            例如: [['', '', '亲', '爱'], ['北', '京']]
        """
        # 清理标记及其内容
        # 1. 删除 <insert>...</insert> 及其内部内容（非贪婪匹配）
        cleaned = re.sub(r'<insert>.*?</insert>', '', ground_truth, flags=re.DOTALL)
        # 2. 删除 <?> 标记
        cleaned = re.sub(r'<\?>', '', cleaned)
        
        # 按换行符分割成行
        lines = cleaned.split('\n')
        
        result = []
        for line in lines:
            # 仅去除行尾空格（保留行首空格）
            line = line.rstrip()
            
            # if not line:  # 跳过空行
            #     continue
            
            # 计算行首空格数量
            leading_spaces = len(line) - len(line.lstrip(' '))
            
            # 去除所有空格后的内容
            line_content = line.replace(' ', '')
            
            # 构建字符列表：行首空格用 '' 表示
            chars = [self.empty_char] * leading_spaces + list(line_content)
            
            result.append(chars)
        
        return result
    
    def prepare_predictions(self, predicted_results: List[Dict]) -> List[List[Dict]]:
        """
        预处理预测结果（现在包含空格子）
        
        参数:
            predicted_results: 完整的识别结果列表（包含空格子）
                [{'row': 0, 'col': 0, 'text': '', 'is_empty': True}, 
                 {'row': 0, 'col': 2, 'text': '親', 'confidence': 0.98, 'is_empty': False}, ...]
        
        返回:
            按行分组的结果 [[第1行结果], [第2行结果], ...]
        """
        # 按 row 分组
        rows_dict = {}
        for result in predicted_results:
            row = result.get('row', 0)
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(result)
        
        # 按 row 排序，每行内按 col 排序
        sorted_rows = []
        for row_idx in sorted(rows_dict.keys()):
            row_results = sorted(rows_dict[row_idx], key=lambda x: x.get('col', 0))
            sorted_rows.append(row_results)
        
        return sorted_rows
    
    def align_rows(
        self, 
        gt_rows: List[List[str]], 
        pred_rows: List[List[Dict]]
    ) -> List[Tuple[List[str], List[Dict]]]:
        """
        对齐 Ground Truth 和预测结果的行
        
        参数:
            gt_rows: Ground Truth 按行分割的字符列表
            pred_rows: 预测结果按行分组的列表
        
        返回:
            对齐后的 (GT行, 预测行) 元组列表
        """
        aligned = []
        
        # 使用较短的长度，避免索引越界
        min_rows = min(len(gt_rows), len(pred_rows))
        
        for i in range(min_rows):
            aligned.append((gt_rows[i], pred_rows[i]))
        
        # 处理不匹配的行
        if len(gt_rows) > len(pred_rows):
            # GT 有更多行
            for i in range(len(pred_rows), len(gt_rows)):
                aligned.append((gt_rows[i], []))
        elif len(pred_rows) > len(gt_rows):
            # 预测有更多行
            for i in range(len(gt_rows), len(pred_rows)):
                aligned.append(([], pred_rows[i]))
        
        return aligned
    
    def extract_text_from_predictions(self, pred_row: List[Dict]) -> List[str]:
        """
        从预测结果中提取文本（现在已经包含空格子）
        
        参数:
            pred_row: 一行的预测结果（已按 col 排序，包含空格子）
        
        返回:
            字符列表（包括空字符串，表示空格子）
        """
        return [result.get('text', self.empty_char) for result in pred_row]
