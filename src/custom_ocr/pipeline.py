"""
后处理管道

实现可组合的后处理管道，支持灵活添加和组合处理器。
"""


class PostProcessingPipeline:
    """
    后处理管道（责任链模式）
    
    允许将多个处理器串联，形成灵活的处理流程。
    每个处理器接收一个数据字典，处理后传递给下一个处理器。
    
    Example:
        >>> pipeline = PostProcessingPipeline(character_list)
        >>> pipeline.add_processor(TopKDecoder(k=5))
        >>> pipeline.add_processor(CTCDeduplicator())
        >>> pipeline.add_processor(ConfidenceFilter(threshold=0.3))
        >>> result = pipeline.process(data)
    """
    
    def __init__(self, character_list):
        """
        初始化管道
        
        Args:
            character_list: 字符映射表（从识别器获取）
        """
        self.character_list = character_list
        self.processors = []
        self._execution_log = []
    
    def add_processor(self, processor):
        """
        添加处理器到管道
        
        Args:
            processor: 处理器实例（需要实现 __call__ 方法）
            
        Returns:
            self: 支持链式调用
        """
        # 将字符映射表注入处理器（如果需要）
        if hasattr(processor, 'set_character_list'):
            processor.set_character_list(self.character_list)
        
        self.processors.append(processor)
        return self
    
    def process(self, data):
        """
        执行完整的处理管道
        
        Args:
            data: 输入数据（通常包含 'prob_matrix'）
            
        Returns:
            dict: 处理后的数据（包含所有中间结果）
        """
        # 确保字符映射表在数据中
        if 'character_list' not in data:
            data['character_list'] = self.character_list
        
        # 重置执行日志
        self._execution_log = []
        
        # 依次执行每个处理器
        for i, processor in enumerate(self.processors):
            processor_name = processor.__class__.__name__
            
            # 执行处理器
            try:
                data = processor(data)
                self._execution_log.append({
                    'step': i + 1,
                    'processor': processor_name,
                    'status': 'success'
                })
            except Exception as e:
                self._execution_log.append({
                    'step': i + 1,
                    'processor': processor_name,
                    'status': 'failed',
                    'error': str(e)
                })
                raise RuntimeError(
                    f"处理器 {processor_name} (步骤 {i+1}) 执行失败: {e}"
                ) from e
        
        # 将执行日志添加到结果中
        data['pipeline_log'] = self._execution_log
        
        return data
    
    def get_processors(self):
        """
        获取当前管道中的所有处理器
        
        Returns:
            list: 处理器列表
        """
        return [
            {
                'index': i,
                'name': processor.__class__.__name__,
                'instance': processor
            }
            for i, processor in enumerate(self.processors)
        ]
    
    def clear(self):
        """
        清空管道中的所有处理器
        """
        self.processors = []
        self._execution_log = []
        return self
    
    def remove_processor(self, index):
        """
        移除指定位置的处理器
        
        Args:
            index: 处理器索引
        """
        if 0 <= index < len(self.processors):
            removed = self.processors.pop(index)
            return removed
        else:
            raise IndexError(f"索引超出范围: {index}")
    
    def get_execution_log(self):
        """
        获取最近一次执行的日志
        
        Returns:
            list: 执行日志
        """
        return self._execution_log
    
    def __len__(self):
        """管道中处理器的数量"""
        return len(self.processors)
    
    def __repr__(self):
        processor_names = [p.__class__.__name__ for p in self.processors]
        return f"PostProcessingPipeline({len(self.processors)} processors: {processor_names})"
