"""
准确率计算器

基于 Grid 的 OCR 识别结果准确率计算
"""

import difflib
import numpy as np
from typing import List, Dict, Optional
from .text_aligner import TextAligner
import re


class GridAccuracyCalculator:
    """基于 Grid 的准确率计算器"""
    
    def __init__(self, empty_char: str = ''):
        """
        初始化计算器
        
        参数:
            empty_char: 空格子的表示字符
        """
        self.empty_char = empty_char
        self.aligner = TextAligner(empty_char)
    
    def calculate(
        self, 
        predicted_results: List[Dict], 
        ground_truth: str,
        align_by_row: bool = True
    ) -> Dict:
        """
        计算准确率
        
        参数:
            predicted_results: 识别结果列表
                [{'row': 0, 'col': 0, 'text': '親', 'confidence': 0.98}, ...]
            ground_truth: Ground Truth 原始字符串（包含换行符）
            align_by_row: 是否按行对齐
        
        返回:
            包含准确率指标的字典
        """
        if not align_by_row:
            # 简单模式：直接拼接对比
            return self._calculate_simple(predicted_results, ground_truth)
        
        # 按行对齐模式
        return self._calculate_by_row(predicted_results, ground_truth)
    
    def _calculate_simple(
        self, 
        predicted_results: List[Dict], 
        ground_truth: str
    ) -> Dict:
        """
        简单模式：直接拼接对比
        """
        # 拼接预测文本
        predicted_text = ''.join([r.get('text', self.empty_char) for r in predicted_results])
        
        # 清理 Ground Truth（简单对齐模式）
        gt_clean = re.sub(r'<insert>.*?</insert>', '', ground_truth, flags=re.DOTALL)
        gt_clean = re.sub(r'<\?>', '', gt_clean)
        gt_clean = gt_clean.replace('\n', '').replace(' ', '')
        
        # 使用 difflib 计算
        matcher = difflib.SequenceMatcher(None, predicted_text, gt_clean)
        similarity = matcher.ratio()
        
        total_chars = len(gt_clean)
        matches = sum(block.size for block in matcher.get_matching_blocks())
        
        # 置信度统计
        confidences = [r.get('confidence', 0) for r in predicted_results]
        
        return {
            'overall': {
                'accuracy': (matches / total_chars * 100) if total_chars > 0 else 0,
                'similarity': similarity * 100,
                'total_chars': total_chars,
                'predicted_chars': len(predicted_text),
                'matched_chars': matches,
                'empty_cells': sum(1 for r in predicted_results if not r.get('text', ''))
            },
            'confidence_stats': {
                'mean': float(np.mean(confidences)) if confidences else 0,
                'min': float(np.min(confidences)) if confidences else 0,
                'max': float(np.max(confidences)) if confidences else 0,
                'low_conf_count': sum(1 for c in confidences if c < 0.5),
            }
        }
    
    def _calculate_by_row(
        self, 
        predicted_results: List[Dict], 
        ground_truth: str
    ) -> Dict:
        """
        按行对齐模式：逐行对比
        """
        # 预处理
        gt_rows = self.aligner.prepare_ground_truth(ground_truth)
        pred_rows = self.aligner.prepare_predictions(predicted_results)
        
        # 对齐
        aligned_rows = self.aligner.align_rows(gt_rows, pred_rows)
        
        # 逐行计算
        by_row_results = []
        total_matches = 0
        total_gt_chars = 0
        total_pred_chars = 0
        
        for row_idx, (gt_row, pred_row) in enumerate(aligned_rows):
            # 提取预测文本（现在是按格子的列表）
            pred_texts = self.aligner.extract_text_from_predictions(pred_row)
            gt_texts = gt_row
            
            # 确保长度一致（补齐较短的）
            max_len = max(len(pred_texts), len(gt_texts))
            pred_texts += [self.empty_char] * (max_len - len(pred_texts))
            gt_texts += [self.empty_char] * (max_len - len(gt_texts))
            
            # 逐格子对比
            cell_matches = sum(1 for p, g in zip(pred_texts, gt_texts) if p == g)
            total_cells = len(gt_texts)
            
            # 转换为字符串（用于显示和错误分析）
            gt_text = ''.join(gt_texts)
            pred_text = ''.join(pred_texts)
            
            # 使用 difflib 计算相似度
            matcher = difflib.SequenceMatcher(None, pred_text, gt_text)
            similarity = matcher.ratio()
            
            # 计算匹配数（字符级别）
            matches = sum(block.size for block in matcher.get_matching_blocks())
            gt_length = len(gt_text)
            pred_length = len(pred_text)
            
            # 记录错误
            errors = []
            opcodes = matcher.get_opcodes()
            for tag, i1, i2, j1, j2 in opcodes:
                if tag != 'equal':
                    errors.append({
                        'type': tag,  # 'replace', 'delete', 'insert'
                        'pred_pos': (i1, i2),
                        'gt_pos': (j1, j2),
                        'pred_text': pred_text[i1:i2] if i1 < len(pred_text) else '',
                        'gt_text': gt_text[j1:j2] if j1 < len(gt_text) else ''
                    })
            
            # 提取置信度（空格子置信度为 1.0）
            confidences = [p.get('confidence', 1.0 if p.get('is_empty', False) else 0) for p in pred_row]
            confidences += [1.0] * (max_len - len(confidences))
            
            # 行结果
            row_result = {
                'row': row_idx,
                'accuracy': (cell_matches / total_cells * 100) if total_cells > 0 else 0,
                'similarity': similarity * 100,
                'gt_text': gt_text,
                'pred_text': pred_text,
                'pred_cells': pred_texts,  # 新增：按格子的列表
                'gt_cells': gt_texts,      # 新增：按格子的列表
                'cell_matches': cell_matches,  # 新增：格子匹配数
                'total_cells': total_cells,    # 新增：总格子数
                'gt_length': gt_length,
                'pred_length': pred_length,
                'matched': matches,
                'errors': errors,
                'confidences': confidences
            }
            
            by_row_results.append(row_result)
            
            total_matches += matches
            total_gt_chars += gt_length
            total_pred_chars += pred_length
        
        # 整体统计
        all_confidences = [r.get('confidence', 0) for r in predicted_results]
        empty_cells = sum(1 for r in predicted_results if not r.get('text', ''))
        
        # 错误分析
        empty_cell_errors = 0
        char_errors = 0
        missing_chars = 0
        extra_chars = 0
        
        for row_result in by_row_results:
            for error in row_result['errors']:
                if error['type'] == 'replace':
                    char_errors += max(len(error['pred_text']), len(error['gt_text']))
                elif error['type'] == 'delete':
                    missing_chars += len(error['gt_text'])
                elif error['type'] == 'insert':
                    extra_chars += len(error['pred_text'])
        
        # 低置信度准确率
        low_conf_results = [r for r in predicted_results if r.get('confidence', 0) < 0.5]
        low_conf_correct = sum(
            1 for r in low_conf_results 
            if r.get('text', '') != self.empty_char
        )
        
        # 计算置信度分布
        conf_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        conf_counts = [0] * (len(conf_bins) - 1)
        conf_error_counts = [0] * (len(conf_bins) - 1)
        
        for pred in predicted_results:
            if pred.get('is_empty', False):
                continue
            conf = pred.get('confidence', 0)
            for i in range(len(conf_bins) - 1):
                if conf_bins[i] <= conf < conf_bins[i + 1]:
                    conf_counts[i] += 1
                    break
        
        # 统计每个区间的错误数（需要重新遍历 by_row）
        for row_result in by_row_results:
            for pred_text, gt_text, conf in zip(
                row_result['pred_cells'], 
                row_result['gt_cells'], 
                row_result['confidences']
            ):
                if pred_text == '':  # 跳过空格子
                    continue
                if pred_text != gt_text:
                    for i in range(len(conf_bins) - 1):
                        if conf_bins[i] <= conf < conf_bins[i + 1]:
                            conf_error_counts[i] += 1
                            break
        
        return {
            'overall': {
                'accuracy': (total_matches / total_gt_chars * 100) if total_gt_chars > 0 else 0,
                'similarity': (total_matches / max(total_gt_chars, total_pred_chars) * 100) if max(total_gt_chars, total_pred_chars) > 0 else 0,
                'total_chars': total_gt_chars,
                'predicted_chars': total_pred_chars,
                'matched_chars': total_matches,
                'empty_cells': empty_cells,
                'total_rows': len(aligned_rows)
            },
            'by_row': by_row_results,
            'confidence_stats': {
                'mean': float(np.mean(all_confidences)) if all_confidences else 0,
                'min': float(np.min(all_confidences)) if all_confidences else 0,
                'max': float(np.max(all_confidences)) if all_confidences else 0,
                'low_conf_count': sum(1 for c in all_confidences if c < 0.5),
                'low_conf_accuracy': (low_conf_correct / len(low_conf_results) * 100) if low_conf_results else 0
            },
            'confidence_distribution': {
                'bins': conf_bins,
                'counts': conf_counts,
                'error_counts': conf_error_counts
            },
            'error_analysis': {
                'empty_cell_errors': empty_cell_errors,
                'char_errors': char_errors,
                'missing_chars': missing_chars,
                'extra_chars': extra_chars,
                'total_errors': total_gt_chars - total_matches
            }
        }
    
    def calculate_with_bert_tracking(
        self,
        original_results: List[Dict],
        enhanced_results: List[Dict],
        ground_truth: str,
        align_by_row: bool = True
    ) -> Dict:
        """
        计算准确率并追踪 BERT 增强的详细效果
        
        参数:
            original_results: 原始 OCR 结果列表
            enhanced_results: BERT 增强后的结果列表
            ground_truth: Ground Truth 原始字符串
            align_by_row: 是否按行对齐
        
        返回:
            包含准确率指标和 BERT 追踪信息的字典
        """
        # 首先计算增强后的准确率（使用现有方法）
        enhanced_metrics = self.calculate(enhanced_results, ground_truth, align_by_row)
        
        # 准备 GT 数据（从 by_row 重构）
        gt_cells = []
        if 'by_row' in enhanced_metrics:
            for row_result in enhanced_metrics['by_row']:
                row_idx = row_result['row']
                for col_idx, gt_text in enumerate(row_result['gt_cells']):
                    gt_cells.append({
                        'row': row_idx,
                        'col': col_idx,
                        'text': gt_text
                    })
        
        # 统计 BERT 改变
        non_empty_count = 0
        triggered_count = 0
        corrected_count = 0  # wrong → correct
        error_introduced_count = 0  # correct → wrong
        wrong_to_wrong_count = 0  # wrong → wrong
        
        bert_changes = []
        confusion_dict = {}  # 用于统计混淆对
        
        for orig, enh, gt in zip(original_results, enhanced_results, gt_cells):
            # 跳过空格子
            if orig.get('is_empty', False):
                continue
            
            non_empty_count += 1
            
            orig_text = orig.get('text', '')
            enh_text = enh.get('text', '')
            gt_text = gt['text']
            
            # 记录混淆对（原始错误）
            if orig_text != gt_text:
                pair = (orig_text, gt_text)
                confusion_dict[pair] = confusion_dict.get(pair, 0) + 1
            
            # 判断是否被 BERT 改变
            if orig_text != enh_text:
                triggered_count += 1
                
                orig_correct = (orig_text == gt_text)
                enh_correct = (enh_text == gt_text)
                
                # 分类改变类型
                if not orig_correct and enh_correct:
                    change_type = 'wrong_to_correct'
                    corrected_count += 1
                elif orig_correct and not enh_correct:
                    change_type = 'correct_to_wrong'
                    error_introduced_count += 1
                elif not orig_correct and not enh_correct:
                    change_type = 'wrong_to_wrong'
                    wrong_to_wrong_count += 1
                else:  # correct_to_correct (不应该发生，但记录以防万一)
                    change_type = 'correct_to_correct'
                
                # 记录详细改变
                bert_changes.append({
                    'row': gt['row'],
                    'col': gt['col'],
                    'original': orig_text,
                    'enhanced': enh_text,
                    'gt': gt_text,
                    'type': change_type,
                    'original_conf': orig.get('confidence', 0),
                    'enhanced_conf': enh.get('confidence', 0)
                })
        
        # 计算统计指标
        trigger_rate = triggered_count / non_empty_count if non_empty_count > 0 else 0
        correction_rate = corrected_count / triggered_count if triggered_count > 0 else 0
        error_introduction_rate = error_introduced_count / triggered_count if triggered_count > 0 else 0
        net_corrections = corrected_count - error_introduced_count
        
        # Top-20 混淆对
        top_confusions = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        top_confusions = [(pred, gt, count) for (pred, gt), count in top_confusions]
        
        # 添加 BERT 追踪信息到结果中
        enhanced_metrics['bert_stats'] = {
            'trigger_rate': trigger_rate,
            'triggered_count': triggered_count,
            'non_empty_count': non_empty_count,
            'correction_rate': correction_rate,
            'corrected_count': corrected_count,
            'error_introduction_rate': error_introduction_rate,
            'error_introduced_count': error_introduced_count,
            'wrong_to_wrong_count': wrong_to_wrong_count,
            'net_corrections': net_corrections
        }
        enhanced_metrics['bert_changes'] = bert_changes
        enhanced_metrics['top_confusions'] = top_confusions
        
        return enhanced_metrics
    
    def generate_report(self, results: Dict, format: str = 'text') -> str:
        """
        生成可读报告
        
        参数:
            results: calculate() 返回的结果
            format: 报告格式 ('text' 或 'markdown')
        
        返回:
            格式化的报告字符串
        """
        if format == 'text':
            return self._generate_text_report(results)
        elif format == 'markdown':
            return self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_text_report(self, results: Dict) -> str:
        """生成文本格式报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("准确率评估报告")
        lines.append("=" * 80)
        
        # 整体统计
        overall = results['overall']
        lines.append(f"\n准确率: {overall['accuracy']:.2f}%")
        lines.append(f"相似度: {overall['similarity']:.2f}%")
        lines.append(f"Ground Truth 长度: {overall['total_chars']} 字符")
        lines.append(f"预测文本长度: {overall['predicted_chars']} 字符")
        lines.append(f"匹配字符数: {overall['matched_chars']}")
        lines.append(f"空格子数: {overall['empty_cells']}")
        
        # 置信度统计
        if 'confidence_stats' in results:
            conf = results['confidence_stats']
            lines.append(f"\n置信度统计:")
            lines.append(f"  平均: {conf['mean']:.3f}")
            lines.append(f"  最低: {conf['min']:.3f}")
            lines.append(f"  最高: {conf['max']:.3f}")
            lines.append(f"  低置信度(<0.5): {conf['low_conf_count']} 个")
        
        # 错误分析
        if 'error_analysis' in results:
            err = results['error_analysis']
            lines.append(f"\n错误分析:")
            lines.append(f"  字符错误: {err['char_errors']}")
            lines.append(f"  漏识别: {err['missing_chars']}")
            lines.append(f"  多识别: {err['extra_chars']}")
            lines.append(f"  总错误: {err['total_errors']}")
        
        # 按行详情（只显示前10行）
        if 'by_row' in results:
            lines.append(f"\n按行准确率（前10行）:")
            lines.append("-" * 80)
            for row_result in results['by_row'][:10]:
                lines.append(f"行 {row_result['row']:2d}: {row_result['accuracy']:5.1f}% | "
                           f"GT: {row_result['gt_text'][:30]:30s} | "
                           f"预测: {row_result['pred_text'][:30]:30s}")
        
        lines.append("=" * 80)
        return '\n'.join(lines)
    
    def _generate_markdown_report(self, results: Dict) -> str:
        """生成 Markdown 格式报告"""
        lines = []
        lines.append("# 准确率评估报告\n")
        
        # 整体统计
        overall = results['overall']
        lines.append("## 整体统计\n")
        lines.append(f"- **准确率**: {overall['accuracy']:.2f}%")
        lines.append(f"- **相似度**: {overall['similarity']:.2f}%")
        lines.append(f"- **Ground Truth 长度**: {overall['total_chars']} 字符")
        lines.append(f"- **预测文本长度**: {overall['predicted_chars']} 字符")
        lines.append(f"- **匹配字符数**: {overall['matched_chars']}")
        lines.append(f"- **空格子数**: {overall['empty_cells']}\n")
        
        # 置信度统计
        if 'confidence_stats' in results:
            conf = results['confidence_stats']
            lines.append("## 置信度统计\n")
            lines.append(f"- 平均: {conf['mean']:.3f}")
            lines.append(f"- 最低: {conf['min']:.3f}")
            lines.append(f"- 最高: {conf['max']:.3f}")
            lines.append(f"- 低置信度(<0.5): {conf['low_conf_count']} 个\n")
        
        return '\n'.join(lines)
