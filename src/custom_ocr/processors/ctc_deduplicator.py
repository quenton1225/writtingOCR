"""
CTC 去重器

实现 CTC 解码的去重逻辑，将时间序列转换为文本。
"""

import numpy as np


class CTCDeduplicator:
    """
    CTC 去重器
    
    实现 Connectionist Temporal Classification (CTC) 的解码逻辑：
    1. 移除连续重复的字符
    2. 移除 blank 标记（通常索引为 0）
    3. 将剩余字符拼接成文本
    
    支持多种去重模式和保留中间信息用于调试。
    """
    
    def __init__(self, blank_idx=0, mode='standard'):
        """
        初始化 CTC 去重器
        
        Args:
            blank_idx: Blank 标记的索引（默认 0）
            mode: 去重模式
                - 'standard': 标准 CTC 去重（移除连续重复和 blank）
                - 'keep_blank': 保留 blank 位置信息
                - 'all': 返回所有中间结果
        """
        self.blank_idx = blank_idx
        self.mode = mode
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
        执行 CTC 解码
        
        Args:
            data: 包含 'top_k_indices' 或 'prob_matrix' 的字典
            
        Returns:
            dict: 添加了以下键的数据字典
                - 'decoded_text': 解码后的文本（列表，每个样本一个）
                - 'decoded_indices': 去重后的字符索引序列
                - 'char_positions': 每个字符在原序列中的位置
                - 'avg_confidence': 平均置信度
        """
        # 获取字符列表
        character_list = data.get('character_list', self.character_list)
        
        # 如果有 top_k_indices，使用 top-1（最高概率）
        if 'top_k_indices' in data:
            indices = data['top_k_indices'][..., 0]  # [batch, time_steps]
            probs = data['top_k_probs'][..., 0]
        else:
            # 否则从概率矩阵中提取 argmax
            prob_matrix = data['prob_matrix']
            indices = prob_matrix.argmax(axis=-1)  # [batch, time_steps]
            probs = prob_matrix.max(axis=-1)
        
        batch_size = indices.shape[0]
        decoded_texts = []
        decoded_indices_list = []
        char_positions_list = []
        avg_confidences = []
        
        for b in range(batch_size):
            sequence = indices[b]
            prob_seq = probs[b]
            
            # 执行 CTC 去重
            decoded_indices, positions, conf = self._decode_sequence(
                sequence, prob_seq
            )
            
            # 转换为文本
            if character_list is not None:
                text = ''.join([
                    character_list[idx] if 0 <= idx < len(character_list) else '<UNK>'
                    for idx in decoded_indices
                ])
            else:
                text = str(decoded_indices)
            
            decoded_texts.append(text)
            decoded_indices_list.append(decoded_indices)
            char_positions_list.append(positions)
            avg_confidences.append(conf)
        
        # 添加到数据中
        data['decoded_text'] = decoded_texts
        data['decoded_indices'] = decoded_indices_list
        data['char_positions'] = char_positions_list
        data['avg_confidence'] = avg_confidences
        
        # 如果只有一个样本，简化输出
        if batch_size == 1:
            data['text'] = decoded_texts[0]
            data['confidence'] = avg_confidences[0]
        
        return data
    
    def _decode_sequence(self, sequence, prob_seq):
        """
        解码单个序列
        
        Args:
            sequence: 字符索引序列 [time_steps]
            prob_seq: 概率序列 [time_steps]
            
        Returns:
            tuple: (decoded_indices, positions, avg_confidence)
                - decoded_indices: 去重后的字符索引列表
                - positions: 每个字符在原序列中的位置
                - avg_confidence: 平均置信度
        """
        decoded_indices = []
        positions = []
        confidences = []
        
        prev_idx = None
        
        for t, (idx, prob) in enumerate(zip(sequence, prob_seq)):
            # 跳过 blank 标记
            if idx == self.blank_idx:
                prev_idx = idx
                continue
            
            # 跳过连续重复（CTC 核心规则）
            if idx == prev_idx:
                continue
            
            # 保留当前字符
            decoded_indices.append(int(idx))
            positions.append(t)
            confidences.append(float(prob))
            
            prev_idx = idx
        
        # 计算平均置信度
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return decoded_indices, positions, avg_confidence
    
    def decode_with_alternatives(self, data, k=3):
        """
        解码时保留前 k 个候选（用于 Beam Search 等）
        
        Args:
            data: 包含 'top_k_indices' 的数据字典
            k: 保留的候选数
            
        Returns:
            list: 每个样本的前 k 个候选文本
        """
        if 'top_k_indices' not in data:
            raise ValueError("需要先运行 TopKDecoder 获取 top-k 候选")
        
        top_k_indices = data['top_k_indices']
        top_k_probs = data['top_k_probs']
        character_list = data.get('character_list', self.character_list)
        
        batch_size = top_k_indices.shape[0]
        k = min(k, top_k_indices.shape[-1])
        
        all_candidates = []
        
        for b in range(batch_size):
            candidates = []
            for i in range(k):
                sequence = top_k_indices[b, :, i]
                prob_seq = top_k_probs[b, :, i]
                decoded_indices, _, conf = self._decode_sequence(sequence, prob_seq)
                
                if character_list is not None:
                    text = ''.join([character_list[idx] for idx in decoded_indices])
                else:
                    text = str(decoded_indices)
                
                candidates.append({
                    'text': text,
                    'confidence': conf,
                    'rank': i + 1
                })
            
            all_candidates.append(candidates)
        
        return all_candidates
    
    def visualize_ctc_alignment(self, data, sample_idx=0):
        """
        可视化 CTC 对齐结果（用于调试）
        
        Args:
            data: 解码后的数据
            sample_idx: 样本索引
            
        Returns:
            str: 可视化字符串
        """
        if 'top_k_indices' not in data:
            return "需要 top_k_indices 数据"
        
        indices = data['top_k_indices'][sample_idx, :, 0]
        probs = data['top_k_probs'][sample_idx, :, 0]
        character_list = data.get('character_list', self.character_list)
        
        lines = []
        lines.append("CTC 对齐可视化:")
        lines.append("-" * 60)
        
        for t, (idx, prob) in enumerate(zip(indices, probs)):
            if character_list is not None and 0 <= idx < len(character_list):
                char = character_list[idx]
                # Blank 用 '_' 表示
                if idx == self.blank_idx:
                    char = '_'
            else:
                char = '?'
            
            bar = '█' * int(prob * 20)
            lines.append(f"t={t:2d}: [{char}] {prob:.3f} {bar}")
        
        lines.append("-" * 60)
        lines.append(f"解码结果: {data['decoded_text'][sample_idx]}")
        
        return '\n'.join(lines)
    
    def __repr__(self):
        return f"CTCDeduplicator(blank_idx={self.blank_idx}, mode='{self.mode}')"
