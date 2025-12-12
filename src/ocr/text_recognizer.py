"""
OCR文字识别模块
基于PaddleOCR实现手写中文识别
"""

from paddleocr import PaddleOCR
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        """返回文字框的中心点"""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return (sum(x_coords) // 4, sum(y_coords) // 4)
    
    @property
    def rect_bbox(self) -> List[int]:
        """返回矩形边界框 [x_min, y_min, x_max, y_max]"""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


class TextRecognizer:
    """手写文字识别器"""
    
    def __init__(
        self, 
        lang: str = "ch"
    ):
        """
        初始化识别器(PaddleOCR 3.x)
        
        Args:
            lang: 语言(ch=中文, en=英文)
        """
        # PaddleOCR 3.x API - 禁用文档级功能以提速
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,  # 不需要文档方向分类
            use_doc_unwarping=False,              # 不需要文档矫正
            use_textline_orientation=False        # 不需要文本行方向检测
        )
        
    def recognize(
        self, 
        image_path: str,
        confidence_threshold: float = 0.5
    ) -> List[OCRResult]:
        """
        识别图片中的文字
        
        Args:
            image_path: 图片路径
            confidence_threshold: 置信度阈值,低于此值的结果将被过滤
            
        Returns:
            OCR识别结果列表
        """
        # 使用 PaddleOCR 3.x 官方 API
        result = self.ocr.predict(input=image_path)
        
        if not result or len(result) == 0:
            return []
        
        ocr_results = []
        # 根据官方文档,result 是结果对象列表
        for res in result:
            # 从结果对象中提取数据
            if hasattr(res, 'boxes') and hasattr(res, 'rec_text') and hasattr(res, 'rec_score'):
                boxes = res.boxes
                texts = res.rec_text
                scores = res.rec_score
                
                for bbox, text, score in zip(boxes, texts, scores):
                    if score >= confidence_threshold:
                        ocr_results.append(OCRResult(
                            text=text,
                            bbox=bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                            confidence=float(score)
                        ))
        
        return ocr_results
    
    def recognize_with_visualization(
        self, 
        image_path: str,
        output_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[OCRResult], np.ndarray]:
        """
        识别文字并生成可视化结果
        
        Args:
            image_path: 图片路径
            output_path: 可视化结果保存路径(可选)
            confidence_threshold: 置信度阈值
            
        Returns:
            (识别结果列表, 可视化图像)
        """
        results = self.recognize(image_path, confidence_threshold)
        
        # 读取原图
        image = cv2.imread(image_path)
        vis_image = image.copy()
        
        # 绘制识别结果
        for result in results:
            bbox = np.array(result.bbox, dtype=np.int32)
            
            # 绘制边界框
            cv2.polylines(vis_image, [bbox], True, (0, 255, 0), 2)
            
            # 添加文字标签
            x, y = result.center
            label = f"{result.text} ({result.confidence:.2f})"
            cv2.putText(
                vis_image, 
                label, 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return results, vis_image
    
    def batch_recognize(
        self,
        image_dir: str,
        pattern: str = "*.png",
        confidence_threshold: float = 0.5
    ) -> Dict[str, List[OCRResult]]:
        """
        批量识别文件夹中的图片
        
        Args:
            image_dir: 图片文件夹路径
            pattern: 文件匹配模式
            confidence_threshold: 置信度阈值
            
        Returns:
            {文件名: 识别结果列表}
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob(pattern))
        
        results = {}
        for img_file in image_files:
            ocr_results = self.recognize(str(img_file), confidence_threshold)
            results[img_file.name] = ocr_results
        
        return results


def extract_plain_text(ocr_results: List[OCRResult]) -> str:
    """
    从OCR结果中提取纯文本
    
    Args:
        ocr_results: OCR识别结果列表
        
    Returns:
        拼接的纯文本
    """
    # 按垂直位置(y坐标)排序
    sorted_results = sorted(ocr_results, key=lambda r: r.center[1])
    
    # 拼接文本
    return ''.join(r.text for r in sorted_results)


def group_by_line(
    ocr_results: List[OCRResult],
    line_height_threshold: int = 30
) -> List[List[OCRResult]]:
    """
    将OCR结果按行分组
    
    Args:
        ocr_results: OCR识别结果列表
        line_height_threshold: 行高阈值,y坐标差距小于此值的视为同一行
        
    Returns:
        按行分组的结果
    """
    if not ocr_results:
        return []
    
    # 按y坐标排序
    sorted_results = sorted(ocr_results, key=lambda r: r.center[1])
    
    lines = []
    current_line = [sorted_results[0]]
    
    for result in sorted_results[1:]:
        # 判断是否与当前行在同一水平线上
        if abs(result.center[1] - current_line[0].center[1]) <= line_height_threshold:
            current_line.append(result)
        else:
            # 按x坐标排序当前行
            current_line.sort(key=lambda r: r.center[0])
            lines.append(current_line)
            current_line = [result]
    
    # 添加最后一行
    if current_line:
        current_line.sort(key=lambda r: r.center[0])
        lines.append(current_line)
    
    return lines
