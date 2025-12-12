"""
自定义文本识别器

封装 PaddleOCR，提供访问底层组件和概率分布的接口。
"""

import numpy as np
from pathlib import Path


class CustomTextRecognizer:
    """
    自定义文本识别器，支持获取完整的概率分布。
    
    与标准 PaddleOCR 的主要区别：
    1. 可以获取原始概率矩阵（而非仅 top-1 结果）
    2. 提供访问底层组件的接口（预处理、推理、字符映射）
    3. 为自定义后处理管道提供基础
    
    Attributes:
        text_rec: PaddleOCR 的 TextRecognition 实例
        predictor: PaddleX 预测器
        character_list: 字符映射表（索引 -> 字符）
    """
    
    def __init__(self, model_name='PP-OCRv5_server_rec', device='gpu:0'):
        """
        初始化识别器
        
        Args:
            model_name: 模型名称，默认使用 PP-OCRv5 服务器版
            device: 设备类型，'gpu:0' 或 'cpu'
        """
        from paddleocr import TextRecognition
        
        # 初始化 PaddleOCR 模型
        self.text_rec = TextRecognition(model_name=model_name, device=device)
        
        # 访问底层预测器
        self.predictor = self.text_rec.paddlex_predictor
        
        # 获取预处理、推理和后处理组件
        self.pre_tfs = self.predictor.pre_tfs
        self.infer = self.predictor.infer
        self.post_op = self.predictor.post_op
        
        # 获取字符映射表
        self.character_list = self.post_op.character
        
        print(f"✓ 加载模型: {model_name}")
        print(f"✓ 字符类别数: {len(self.character_list)}")
        print(f"✓ 设备: {device}")
    
    def predict_with_raw_output(self, img_path):
        """
        识别文本并返回原始概率矩阵
        
        Args:
            img_path: 图像路径（字符串或 Path 对象）或 NumPy 数组
            
        Returns:
            dict: 包含以下键
                - 'prob_matrix': 概率矩阵 [batch, time_steps, num_classes]
                - 'raw_image': 原始图像（预处理前）
                - 'character_list': 字符映射表
                - 'image_shape': 图像形状
        """
        # 预处理
        if isinstance(img_path, (str, Path)):
            batch_raw_imgs = self.pre_tfs["Read"](imgs=[str(img_path)])
        elif isinstance(img_path, np.ndarray):
            batch_raw_imgs = [img_path]
        else:
            raise TypeError(f"不支持的输入类型: {type(img_path)}")
        
        # 图像调整大小和归一化
        batch_imgs = self.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
        
        # 转为批次格式
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        
        # 推理（获取原始输出）
        batch_preds = self.infer(x=x)
        
        # batch_preds[0] 是完整的概率矩阵
        # 形状: [batch_size, time_steps, num_classes]
        prob_matrix = np.array(batch_preds[0])
        
        return {
            'prob_matrix': prob_matrix,
            'raw_image': batch_raw_imgs[0],
            'character_list': self.character_list,
            'image_shape': batch_raw_imgs[0].shape,
            'batch_size': prob_matrix.shape[0],
            'time_steps': prob_matrix.shape[1],
            'num_classes': prob_matrix.shape[2],
        }
    
    def predict_standard(self, img_path):
        """
        标准预测接口（向后兼容）
        
        Args:
            img_path: 图像路径
            
        Returns:
            list: PaddleOCR 标准格式的结果
        """
        return self.text_rec.predict([img_path])
    
    def get_character_list(self):
        """
        获取字符映射表
        
        Returns:
            list: 字符列表，索引对应字符类别
        """
        return self.character_list
    
    def get_character_by_index(self, index):
        """
        根据索引获取字符
        
        Args:
            index: 字符索引
            
        Returns:
            str: 对应字符
        """
        if 0 <= index < len(self.character_list):
            return self.character_list[index]
        else:
            raise IndexError(f"索引超出范围: {index} (总数: {len(self.character_list)})")
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 模型配置信息
        """
        return {
            'model_name': self.predictor.model_name,
            'num_characters': len(self.character_list),
            'blank_index': 0,  # CTC blank 标记的索引
            'preprocessors': list(self.pre_tfs.keys()),
        }
    
    def batch_predict_with_raw_output(self, img_paths):
        """
        批量识别（返回原始输出）
        
        Args:
            img_paths: 图像路径列表
            
        Returns:
            list: 每张图像的结果字典
        """
        results = []
        for img_path in img_paths:
            result = self.predict_with_raw_output(img_path)
            results.append(result)
        return results
    
    def __repr__(self):
        return (f"CustomTextRecognizer("
                f"model={self.predictor.model_name}, "
                f"num_chars={len(self.character_list)})")
