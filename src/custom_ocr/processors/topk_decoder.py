"""
Top-K 解码器

从概率矩阵中提取每个时间步的 Top-K 候选字符。
"""

import numpy as np


class TopKDecoder:
    """
    Top-K 解码器
    
    从 CTC 输出的概率矩阵中提取每个时间步的 Top-K 候选字符及其概率。
    这是实现上下文增强等高级功能的基础。
    
    Attributes:
        k: 每个时间步保留的候选数
        return_scores: 是否返回归一化的概率分数
    """
    
    def __init__(self, k=5, return_scores=True):
        """
        初始化 Top-K 解码器
        
        Args:
            k: 每个时间步保留的候选数（默认 5）
            return_scores: 是否返回归一化概率（默认 True）
        """
        self.k = k
        self.return_scores = return_scores
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
        执行 Top-K 解码
        
        Args:
            data: 包含 'prob_matrix' 的字典
            
        Returns:
            dict: 添加了以下键的数据字典
                - 'top_k_indices': Top-K 字符索引 [batch, time_steps, k]
                - 'top_k_probs': Top-K 概率值 [batch, time_steps, k]
                - 'top_k_chars': Top-K 字符（如果有字符映射表）
        """
        prob_matrix = data['prob_matrix']
        
        # 获取字符列表（优先使用数据中的）
        character_list = data.get('character_list', self.character_list)
        
        # 检查 k 是否超出类别数
        num_classes = prob_matrix.shape[-1]
        k = min(self.k, num_classes)
        
        if k < self.k:
            print(f"警告: k={self.k} 超过类别数 {num_classes}，自动调整为 {k}")
        
        # 方法 1: 使用 argsort（适合小 k）
        # 获取每个位置概率最高的 k 个索引
        if k <= 10:
            # argsort 返回从小到大的索引，我们取最后 k 个（最大的）
            top_k_indices = np.argsort(prob_matrix, axis=-1)[..., -k:]
            # 反转顺序，使最高概率在前
            top_k_indices = top_k_indices[..., ::-1]
        else:
            # 方法 2: 使用 argpartition（适合大 k）
            # 更快，但结果不保证完全排序
            top_k_indices = np.argpartition(prob_matrix, -k, axis=-1)[..., -k:]
            # 需要再次排序
            batch_indices = np.arange(prob_matrix.shape[0])[:, None, None]
            time_indices = np.arange(prob_matrix.shape[1])[None, :, None]
            k_probs = prob_matrix[batch_indices, time_indices, top_k_indices]
            sort_indices = np.argsort(-k_probs, axis=-1)  # 降序
            top_k_indices = np.take_along_axis(top_k_indices, sort_indices, axis=-1)
        
        # 提取对应的概率值
        batch_indices = np.arange(prob_matrix.shape[0])[:, None, None]
        time_indices = np.arange(prob_matrix.shape[1])[None, :, None]
        top_k_probs = prob_matrix[batch_indices, time_indices, top_k_indices]
        
        # 可选：归一化概率（使 top-k 概率之和为 1）
        if self.return_scores:
            top_k_probs = top_k_probs / (top_k_probs.sum(axis=-1, keepdims=True) + 1e-10)
        
        # 添加到数据中
        data['top_k_indices'] = top_k_indices
        data['top_k_probs'] = top_k_probs
        data['top_k'] = k  # 实际使用的 k 值
        
        # 如果有字符映射表，转换为字符
        if character_list is not None:
            top_k_chars = self._indices_to_chars(top_k_indices, character_list)
            data['top_k_chars'] = top_k_chars
        
        return data
    
    def _indices_to_chars(self, indices, character_list):
        """
        将索引转换为字符
        
        Args:
            indices: 字符索引数组 [batch, time_steps, k]
            character_list: 字符映射表
            
        Returns:
            list: 字符列表 [batch][time_step][k]
        """
        batch_size, time_steps, k = indices.shape
        chars = []
        
        for b in range(batch_size):
            batch_chars = []
            for t in range(time_steps):
                time_chars = []
                for i in range(k):
                    idx = indices[b, t, i]
                    if 0 <= idx < len(character_list):
                        time_chars.append(character_list[idx])
                    else:
                        time_chars.append('<UNK>')
                batch_chars.append(time_chars)
            chars.append(batch_chars)
        
        return chars
    
    def get_top1(self, data):
        """
        从 Top-K 结果中提取 Top-1
        
        Args:
            data: 包含 'top_k_indices' 的数据字典
            
        Returns:
            tuple: (top1_indices, top1_probs)
        """
        if 'top_k_indices' not in data:
            raise ValueError("数据中没有 'top_k_indices'，请先运行 TopKDecoder")
        
        top1_indices = data['top_k_indices'][..., 0]
        top1_probs = data['top_k_probs'][..., 0]
        
        return top1_indices, top1_probs
    
    def __repr__(self):
        return f"TopKDecoder(k={self.k}, return_scores={self.return_scores})"
