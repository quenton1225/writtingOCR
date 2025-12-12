"""
端到端OCR识别管线
原图 → 裁剪 → 格子检测 → OCR识别 → BERT增强 → 评估
"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# 导入现有模块 (不修改它们)
from ..preprocessing.grid_detection import detect_grid_lines, generate_grid_cells
from ..custom_ocr import CustomTextRecognizer
from ..custom_ocr.processors import TopKDecoder, CTCDeduplicator, ConfidenceFilter
from ..semantic import GridContextEnhancer
from ..evaluation import GridAccuracyCalculator
from .image_cropper import auto_crop


class EndToEndOCRPipeline:
    """
    端到端OCR识别管线
    
    整合了从原图到最终识别结果的完整流程:
    1. 图像裁剪
    2. 格子检测与分割
    3. OCR识别
    4. BERT后处理增强
    5. 准确率评估
    
    所有核心逻辑复用现有模块,不修改源代码
    """
    
    def __init__(
        self,
        model_name='PP-OCRv5_server_rec',
        device='gpu:0',
        confidence_threshold=0.90,
        bert_weight=0.55,
        context_window=10,
        verbose=False
    ):
        """
        初始化Pipeline
        
        参数:
            model_name: OCR模型名称
            device: 设备 ('gpu:0' 或 'cpu')
            confidence_threshold: BERT介入的置信度阈值
            bert_weight: BERT权重 (0-1)
            context_window: BERT上下文窗口大小
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        
        # 初始化OCR识别器
        if verbose:
            print("初始化OCR识别器...")
        self.recognizer = CustomTextRecognizer(
            model_name=model_name,
            device=device
        )
        
        # 初始化后处理器 (沿用02.8的配置)
        self.decoder = TopKDecoder(k=5)
        self.deduplicator = CTCDeduplicator()
        self.conf_filter = ConfidenceFilter(threshold=0.0)
        
        # 初始化BERT增强器 (使用最优参数)
        if verbose:
            print("初始化BERT增强器...")
        self.bert_enhancer = GridContextEnhancer(
            model_name='bert-base-chinese',
            device=device,
            context_window=context_window,
            confidence_threshold=confidence_threshold,
            bert_weight=bert_weight,
            verbose=False
        )
        
        # 初始化评估器
        self.calculator = GridAccuracyCalculator(empty_char='')
        
        if verbose:
            print("✓ Pipeline初始化完成")
    
    def process(self, raw_img_path, ground_truth=None, skip_crop=False):
        """
        完整处理流程
        
        参数:
            raw_img_path: 原始图像路径
            ground_truth: Ground Truth文本 (可选,用于评估)
            skip_crop: 是否跳过裁剪 (如果输入已经是裁剪后的图像)
        
        返回:
            dict: 包含以下键值
                - predictions: 完整预测结果列表
                - metrics: 评估指标 (如果提供GT)
                - num_cells: 总格子数
                - num_non_empty: 非空格子数
                - cropped_image: 裁剪后的图像 (PIL.Image)
        """
        if self.verbose:
            print("="*80)
            print("开始端到端OCR识别")
            print("="*80)
        
        # Step 1: 图像裁剪
        if not skip_crop:
            if self.verbose:
                print("\nStep 1: 裁剪图像...")
            cropped_img = auto_crop(raw_img_path)
            if self.verbose:
                print(f"✓ 裁剪完成: {cropped_img.size}")
        else:
            if self.verbose:
                print("\nStep 1: 跳过裁剪,直接加载...")
            cropped_img = Image.open(raw_img_path)
        
        # 转换为OpenCV格式
        img_array = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        
        # Step 2: 格子检测
        if self.verbose:
            print("\nStep 2: 格子检测...")
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        lines = detect_grid_lines(gray)
        cells = generate_grid_cells(
            lines['horizontal'],
            lines['vertical'],
            img_array.shape[:2]
        )
        if self.verbose:
            print(f"✓ 检测到格子: {len(cells)} 个")
        
        # Step 3: 提取非空格子
        if self.verbose:
            print("\nStep 3: 过滤非空格子...")
        cell_images, non_empty_cells = self._extract_non_empty_cells(
            img_array, cells
        )
        if self.verbose:
            print(f"✓ 非空格子: {len(non_empty_cells)} 个")
        
        # Step 4: OCR识别 + 后处理
        if self.verbose:
            print("\nStep 4: OCR识别...")
        results_list = self._ocr_recognize(cell_images)
        if self.verbose:
            print(f"✓ 识别完成: {len(results_list)} 个格子")
        
        # Step 5: BERT增强
        if self.verbose:
            print("\nStep 5: BERT增强...")
        enhanced_results = self.bert_enhancer.enhance_grids(
            grid_results=results_list,
            grid_indices=None
        )
        
        # 统计修正数量
        changed = sum(1 for orig, enh in zip(results_list, enhanced_results)
                     if orig.get('text', '') != enh.get('text', ''))
        if self.verbose:
            print(f"✓ BERT增强完成: 修正了 {changed} 个格子")
        
        # Step 6: 恢复空格子
        if self.verbose:
            print("\nStep 6: 恢复空格子...")
        prediction_data_full = self._restore_empty_cells(
            enhanced_results, cells, non_empty_cells
        )
        if self.verbose:
            print(f"✓ 完整预测结果: {len(prediction_data_full)} 个格子")
        
        # Step 7: 评估 (如果提供GT)
        metrics = None
        if ground_truth is not None:
            if self.verbose:
                print("\nStep 7: 评估准确率...")
            metrics = self.calculator.calculate(
                predicted_results=prediction_data_full,
                ground_truth=ground_truth,
                align_by_row=True
            )
            if self.verbose:
                accuracy = metrics['overall']['accuracy']
                matched = metrics['overall']['matched_chars']
                total = metrics['overall']['total_chars']
                print(f"✓ 字符准确率: {accuracy:.2f}% ({matched}/{total})")
        
        if self.verbose:
            print("\n" + "="*80)
            print("识别完成")
            print("="*80)
        
        return {
            'predictions': prediction_data_full,
            'metrics': metrics,
            'num_cells': len(cells),
            'num_non_empty': len(non_empty_cells),
            'cropped_image': cropped_img
        }
    
    def _extract_non_empty_cells(self, img, cells):
        """
        提取非空格子 (沿用02.8的逻辑)
        """
        cell_images = []
        non_empty_cells = []
        
        for cell in cells:
            x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
            cell_img = img[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                
                # 裁剪中心区域 (避免边框干扰)
                h, w = gray_cell.shape
                crop = int(min(h, w) * 0.08)
                if h > 2*crop and w > 2*crop:
                    center_region = gray_cell[crop:h-crop, crop:w-crop]
                else:
                    center_region = gray_cell
                
                # 计算非白色像素比例
                non_white_ratio = (center_region < 240).sum() / center_region.size
                if non_white_ratio > 0.005:
                    cell_images.append(cell_img)
                    non_empty_cells.append(cell)
        
        return cell_images, non_empty_cells
    
    def _ocr_recognize(self, cell_images):
        """
        OCR识别 + 后处理 (沿用02.8的方法链)
        """
        # 批量获取原始概率矩阵
        batch_raw_outputs = self.recognizer.batch_predict_with_raw_output(cell_images)
        
        # 后处理: Top-K解码 → CTC去重 → 置信度过滤
        results_list = []
        for raw_output in batch_raw_outputs:
            decoded = self.decoder(raw_output)
            deduped = self.deduplicator(decoded)
            filtered = self.conf_filter(deduped)
            results_list.append(filtered)
        
        return results_list
    
    def _restore_empty_cells(self, results_list, cells, non_empty_cells):
        """
        恢复空格子 (沿用02.8的逻辑)
        """
        all_cells_dict = {}
        
        # 初始化所有格子为空
        for cell in cells:
            row, col = cell['row'], cell['col']
            all_cells_dict[(row, col)] = {
                'row': row,
                'col': col,
                'text': '',
                'confidence': 1.0,
                'is_empty': True
            }
        
        # 填充非空格子的识别结果
        for result, cell in zip(results_list, non_empty_cells):
            row, col = cell['row'], cell['col']
            all_cells_dict[(row, col)] = {
                'row': row,
                'col': col,
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0),
                'is_empty': False,
                'ocr_result': result
            }
        
        # 转换为列表并排序
        prediction_data_full = sorted(
            all_cells_dict.values(),
            key=lambda x: (x['row'], x['col'])
        )
        
        return prediction_data_full
