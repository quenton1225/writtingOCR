"""
置信度过滤器

根据置信度过滤或标记低质量字符。
"""

import numpy as np


class ConfidenceFilter:
    """
    置信度过滤器
    
    根据置信度阈值过滤或标记低质量字符，
    为后续的上下文增强等操作标识需要处理的位置。
    
    支持多种过滤策略：
    - remove: 删除低置信度字符
    - mark: 用特殊标记替换
    - flag: 仅标记位置，不修改文本
    """
    
    def __init__(self, threshold=0.3, strategy='flag'):
        """
        初始化置信度过滤器
        
        Args:
            threshold: 置信度阈值（默认 0.3）
            strategy: 过滤策略
                - 'remove': 删除低置信度字符
                - 'mark': 用 <UNK> 标记替换
                - 'flag': 仅添加标志，不修改文本（推荐）
        """
        self.threshold = threshold
        self.strategy = strategy
        self.character_list = None
    
    def set_character_list(self, character_list):
        """
        设置字符映射表（由 Pipeline 自动调用）
        
        Args:
            character_list: 字符映射表
        """
        self.character_list = character_list
    
    def __call__(self, data):
        """
        执行置信度过滤
        
        Args:
            data: 包含 'decoded_text' 和置信度信息的字典
            
        Returns:
            dict: 添加了以下键的数据字典
                - 'filtered_text': 过滤后的文本（如果 strategy 不是 'flag'）
                - 'low_confidence_positions': 低置信度字符的位置列表
                - 'confidence_flags': 每个字符的置信度标志
                - 'num_low_confidence': 低置信度字符数量
        """
        # 获取解码后的文本和置信度信息
        decoded_texts = data.get('decoded_text', [])
        
        # 获取字符级别的置信度
        char_confidences = self._extract_char_confidences(data)
        
        filtered_texts = []
        all_low_conf_positions = []
        all_confidence_flags = []
        num_low_confidence_total = 0
        
        for text, confidences in zip(decoded_texts, char_confidences):
            # 识别低置信度位置
            low_conf_positions = [
                i for i, conf in enumerate(confidences)
                if conf < self.threshold
            ]
            
            # 创建置信度标志
            confidence_flags = [
                'high' if conf >= self.threshold else 'low'
                for conf in confidences
            ]
            
            # 根据策略处理文本
            if self.strategy == 'remove':
                # 删除低置信度字符
                filtered_text = ''.join([
                    char for i, char in enumerate(text)
                    if i not in low_conf_positions
                ])
            elif self.strategy == 'mark':
                # 用 <UNK> 替换
                filtered_text = ''.join([
                    '<UNK>' if i in low_conf_positions else char
                    for i, char in enumerate(text)
                ])
            else:  # 'flag'
                # 不修改文本
                filtered_text = text
            
            filtered_texts.append(filtered_text)
            all_low_conf_positions.append(low_conf_positions)
            all_confidence_flags.append(confidence_flags)
            num_low_confidence_total += len(low_conf_positions)
        
        # 添加到数据中
        data['filtered_text'] = filtered_texts
        data['low_confidence_positions'] = all_low_conf_positions
        data['confidence_flags'] = all_confidence_flags
        data['num_low_confidence'] = num_low_confidence_total
        
        # 如果只有一个样本，简化输出
        if len(decoded_texts) == 1:
            data['filtered_text_single'] = filtered_texts[0]
            data['low_conf_positions_single'] = all_low_conf_positions[0]
        
        # 统计信息
        total_chars = sum(len(text) for text in decoded_texts)
        data['low_confidence_ratio'] = (
            num_low_confidence_total / total_chars
            if total_chars > 0 else 0.0
        )
        
        return data
    
    def _extract_char_confidences(self, data):
        """
        提取每个字符的置信度
        
        Args:
            data: 包含解码结果的数据字典
            
        Returns:
            list: 每个样本的字符级置信度列表
        """
        char_confidences = []
        
        # 如果有 char_positions，使用原始时间步的概率
        if 'char_positions' in data and 'top_k_probs' in data:
            top_k_probs = data['top_k_probs']
            char_positions_list = data['char_positions']
            
            for b, positions in enumerate(char_positions_list):
                confidences = [
                    float(top_k_probs[b, pos, 0])
                    for pos in positions
                ]
                char_confidences.append(confidences)
        
        # 否则使用平均置信度（不够精确）
        elif 'avg_confidence' in data:
            avg_confidences = data['avg_confidence']
            decoded_texts = data['decoded_text']
            
            for text, avg_conf in zip(decoded_texts, avg_confidences):
                # 所有字符使用相同的平均置信度
                confidences = [avg_conf] * len(text)
                char_confidences.append(confidences)
        
        else:
            # 没有置信度信息，使用 1.0（假设都是高置信度）
            decoded_texts = data['decoded_text']
            for text in decoded_texts:
                confidences = [1.0] * len(text)
                char_confidences.append(confidences)
        
        return char_confidences
    
    def get_statistics(self, data):
        """
        获取置信度统计信息
        
        Args:
            data: 已过滤的数据
            
        Returns:
            dict: 统计信息
        """
        if 'low_confidence_positions' not in data:
            return {}
        
        total_chars = sum(len(text) for text in data['decoded_text'])
        low_conf_chars = data['num_low_confidence']
        
        return {
            'total_characters': total_chars,
            'low_confidence_characters': low_conf_chars,
            'low_confidence_ratio': data['low_confidence_ratio'],
            'threshold': self.threshold,
            'strategy': self.strategy,
        }
    
    def visualize_confidence(self, data, sample_idx=0):
        """
        可视化置信度（用于调试）
        
        Args:
            data: 过滤后的数据
            sample_idx: 样本索引
            
        Returns:
            str: 可视化字符串
        """
        if sample_idx >= len(data['decoded_text']):
            return "样本索引超出范围"
        
        text = data['decoded_text'][sample_idx]
        flags = data['confidence_flags'][sample_idx]
        
        # 提取字符级置信度
        char_confs = self._extract_char_confidences(data)[sample_idx]
        
        lines = []
        lines.append("置信度可视化:")
        lines.append("-" * 60)
        lines.append(f"阈值: {self.threshold}")
        lines.append("-" * 60)
        
        for i, (char, flag, conf) in enumerate(zip(text, flags, char_confs)):
            marker = '❌' if flag == 'low' else '✓'
            bar = '█' * int(conf * 20)
            lines.append(f"{i:2d}: [{char}] {conf:.3f} {bar} {marker}")
        
        lines.append("-" * 60)
        
        low_count = sum(1 for f in flags if f == 'low')
        lines.append(f"低置信度字符: {low_count}/{len(text)} ({low_count/len(text)*100:.1f}%)")
        
        return '\n'.join(lines)
    
    def __repr__(self):
        return (f"ConfidenceFilter(threshold={self.threshold}, "
                f"strategy='{self.strategy}')")
