"""
GridContextEnhancer - 格子级 BERT 上下文增强处理器

与 ContextEnhancer 的区别：
- ContextEnhancer: 在 Pipeline 中运行，逐格子处理，字符级 MASK
- GridContextEnhancer: 独立使用，批量处理，格子级 MASK，利用跨格子上下文

核心思想：
将整个格子内容替换为 [MASK]，利用前后格子的识别结果作为上下文，
让 BERT 预测该格子应该是什么字符（支持单字符和多字符预测）。

Author: OCR Pipeline Team
Date: 2025-11-27
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import torch
    from transformers import BertTokenizer, BertForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  警告: transformers 未安装，GridContextEnhancer 将无法使用")
    print("   安装命令: pip install transformers torch")


class GridContextEnhancer:
    """
    格子级上下文增强处理器
    
    使用 BERT 完形填空模型，以整个格子为单位进行增强，充分利用跨格子的上下文信息。
    
    工作原理:
    1. 对于每个低置信度格子，将整个格子内容替换为 [MASK]
    2. 使用前后若干格子的识别结果作为上下文
    3. BERT 预测 [MASK] 位置应该是什么字符（可能是单字或多字）
    4. 融合 OCR 和 BERT 的概率: P_final = α × P_ocr + (1-α) × P_bert
    5. 选择融合分数最高的候选
    
    参数:
        model_name: BERT 模型名称，默认 'bert-base-chinese'
        context_window: 上下文窗口大小（前后各取 N 个格子），默认 7
        fusion_weight: OCR 概率权重（0-1），默认 0.6 (60% OCR + 40% BERT)
        confidence_threshold: 仅增强低于该置信度的格子，默认 0.8
        delta_threshold: Delta回退阈值，仅当best_score > original_score + delta时接受BERT修正，默认 0.15
        correction_threshold: (已废弃，请使用delta_threshold) 向后兼容参数
        device: 计算设备 ('cpu', 'cuda', 'cuda:0' 等)，默认自动检测
        verbose: 是否打印详细调试信息，默认 False
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-chinese',
        context_window: int = 7,
        fusion_weight: float = 0.6,
        confidence_threshold: float = 0.8,
        delta_threshold: Optional[float] = None,
        correction_threshold: Optional[float] = None,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers 库未安装。请运行: pip install transformers torch"
            )
        
        # 参数验证
        if not 0 <= fusion_weight <= 1:
            raise ValueError(f"fusion_weight 必须在 [0, 1] 范围内，当前: {fusion_weight}")
        
        if context_window < 1:
            raise ValueError(f"context_window 必须 >= 1，当前: {context_window}")
        
        # 向后兼容: delta_threshold 优先于 correction_threshold
        if delta_threshold is None and correction_threshold is None:
            delta_threshold = 0.15  # 默认值
        elif delta_threshold is None:
            delta_threshold = correction_threshold
        
        # 保存配置
        self.model_name = model_name
        self.context_window = context_window
        self.fusion_weight = fusion_weight
        self.confidence_threshold = confidence_threshold
        self.delta_threshold = delta_threshold
        self.correction_threshold = delta_threshold  # 保持向后兼容
        self.verbose = verbose
        
        # 设备配置
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 延迟加载模型（首次使用时加载）
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
        
        print(f"✓ GridContextEnhancer 初始化完成")
        print(f"  模型: {model_name}")
        print(f"  上下文窗口: ±{context_window} 个格子")
        print(f"  融合权重: {fusion_weight:.1%} OCR + {1-fusion_weight:.1%} BERT")
        print(f"  设备: {self.device}")
    
    def _load_model(self):
        """延迟加载 BERT 模型"""
        if self._model_loaded:
            return
        
        print(f"正在加载 BERT 模型: {self.model_name}...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            self._model_loaded = True
            print(f"✓ BERT 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"BERT 模型加载失败: {e}")
    
    def enhance_grids(
        self,
        grid_results: List[Dict],
        grid_indices: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        批量增强多个格子
        
        Args:
            grid_results: 所有格子的 Pipeline 处理结果列表
                每个元素是一个 Dict，包含 'text', 'confidence' 等字段
            grid_indices: 需要增强的格子索引列表（None=自动识别低置信度格子）
        
        Returns:
            增强后的结果列表（与输入相同长度）
            在原有字段基础上添加:
                - grid_bert_correction: Dict - 修正详情
                - grid_bert_candidates: List[Tuple] - 所有候选及评分
        """
        # 延迟加载模型
        self._load_model()
        
        # 提取所有格子的文本和置信度
        all_texts = [r.get('text', '') for r in grid_results]
        all_confidences = [r.get('confidence', 0.0) for r in grid_results]
        
        # 确定需要增强的格子
        if grid_indices is None:
            grid_indices = [
                i for i, conf in enumerate(all_confidences)
                if conf < self.confidence_threshold
            ]
            print(f"\n自动识别到 {len(grid_indices)} 个低置信度格子需要增强")
        else:
            print(f"\n将增强指定的 {len(grid_indices)} 个格子")
        
        # 复制结果（避免修改原数据）
        enhanced_results = [r.copy() for r in grid_results]
        
        # 对每个需要增强的格子进行处理
        for idx in grid_indices:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"处理格子 {idx + 1}/{len(grid_results)}")
                print(f"{'='*80}")
            
            enhanced_text, candidates, correction_info = self.enhance_single_grid(
                grid_index=idx,
                all_grid_texts=all_texts,
                all_grid_confidences=all_confidences,
                current_grid_data=grid_results[idx]
            )
            
            # 更新结果
            enhanced_results[idx]['text'] = enhanced_text
            enhanced_results[idx]['grid_bert_correction'] = correction_info
            enhanced_results[idx]['grid_bert_candidates'] = candidates
            
            # 更新置信度：使用融合后的最佳分数
            if correction_info.get('corrected', False):
                # 如果发生修正,使用新的融合分数
                enhanced_results[idx]['confidence'] = correction_info.get('new_score', all_confidences[idx])
            # 如果未修正,保持原始OCR置信度不变
        
        # 统计
        corrected_count = sum(
            1 for r in enhanced_results 
            if r.get('grid_bert_correction', {}).get('corrected', False)
        )
        print(f"\n✓ 格子级增强完成: 检查了 {len(grid_indices)} 个格子，修正了 {corrected_count} 个")
        
        return enhanced_results
    
    def enhance_single_grid(
        self,
        grid_index: int,
        all_grid_texts: List[str],
        all_grid_confidences: List[float],
        current_grid_data: Dict
    ) -> Tuple[str, List[Tuple[str, float, Dict]], Dict]:
        """
        增强单个格子
        
        Args:
            grid_index: 当前格子索引
            all_grid_texts: 所有格子的 OCR 文本
            all_grid_confidences: 所有格子的置信度
            current_grid_data: 当前格子的完整数据（包含 top_k_candidates 等）
        
        Returns:
            (增强后的文本, 所有候选及评分, 修正详情)
        """
        ocr_text = all_grid_texts[grid_index]
        ocr_conf = all_grid_confidences[grid_index]
        
        # 构建跨格子上下文
        left_start = max(0, grid_index - self.context_window)
        right_end = min(len(all_grid_texts), grid_index + self.context_window + 1)
        
        left_context = ''.join(all_grid_texts[left_start:grid_index])
        right_context = ''.join(all_grid_texts[grid_index + 1:right_end])
        
        masked_input = left_context + '[MASK]' + right_context
        
        if self.verbose:
            print(f"OCR 文本: '{ocr_text}' (置信度: {ocr_conf:.3f})")
            print(f"左上下文 (最后20字): '...{left_context[-20:]}'")
            print(f"右上下文 (前20字): '{right_context[:20]}...'")
            print(f"BERT 输入 (最后40字): '...{masked_input[-40:]}'")
        
        # BERT 预测
        bert_predictions = self._predict_bert_for_grid(masked_input)
        
        if self.verbose and bert_predictions:
            print(f"\nBERT 预测 Top-10:")
            for i, (text, prob) in enumerate(list(bert_predictions.items())[:10]):
                char_len = len(text)
                print(f"  {i+1:2d}. '{text}' ({char_len}字) prob={prob:.4f}")
        
        # 提取 OCR 候选（如果有）
        ocr_candidates = current_grid_data.get('top_k_candidates', [])
        
        # 评分所有候选
        scored_candidates = self._score_grid_candidates(
            ocr_text, ocr_conf, bert_predictions, ocr_candidates
        )
        
        if self.verbose and scored_candidates:
            print(f"\n融合后 Top-5 候选:")
            for i, (text, score, details) in enumerate(scored_candidates[:5]):
                print(f"  {i+1}. '{text}' score={score:.3f} "
                      f"(OCR:{details['ocr_prob']:.3f} BERT:{details['bert_prob']:.3f} "
                      f"src:{details['source']})")
        
        # 决策：是否修正
        if not scored_candidates:
            return ocr_text, [], {'corrected': False, 'reason': 'no_candidates'}
        
        best_text, best_score, best_details = scored_candidates[0]
        original_score = ocr_conf  # 使用OCR原始置信度作为基准
        
        should_correct = (
            best_text != ocr_text and 
            best_score > original_score + self.delta_threshold
        )
        
        if should_correct:
            correction_info = {
                'corrected': True,
                'original': ocr_text,
                'corrected_to': best_text,
                'original_score': original_score,
                'new_score': best_score,
                'improvement': best_score - original_score,
                'details': best_details
            }
            if self.verbose:
                improvement = best_score - original_score
                print(f"\n✓ 修正: '{ocr_text}' → '{best_text}' "
                      f"(Δ={improvement:.3f} > δ={self.delta_threshold})")
            return best_text, scored_candidates, correction_info
        else:
            improvement = best_score - original_score
            if self.verbose:
                print(f"\n→ 保持: '{ocr_text}' "
                      f"(最佳'{best_text}' Δ={improvement:.3f} ≤ δ={self.delta_threshold})")
            return ocr_text, scored_candidates, {
                'corrected': False, 
                'reason': 'insufficient_improvement',
                'best_candidate': best_text,
                'best_score': best_score,
                'improvement': improvement,
                'delta_threshold': self.delta_threshold
            }
    
    def _predict_bert_for_grid(self, masked_input: str) -> Dict[str, float]:
        """
        对整个格子进行 BERT 预测，保留所有长度的预测
        
        Args:
            masked_input: 带 [MASK] 的上下文文本
        
        Returns:
            {预测文本: 概率} 字典，可能包含单字、双字、多字预测
            例如: {'：': 0.85, '的': 0.05, '中国': 0.03, ...}
        """
        if not masked_input or '[MASK]' not in masked_input:
            return {}
        
        try:
            inputs = self.tokenizer(
                masked_input,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
            
            # 找到 [MASK] 位置
            mask_token_index = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            
            if len(mask_token_index) == 0:
                return {}
            
            # 获取 [MASK] 位置的预测概率
            mask_token_logits = predictions[0, mask_token_index[0], :]
            probs = torch.softmax(mask_token_logits, dim=-1)
            
            # 获取 Top-K 预测（扩大到 50 个候选）
            top_k = min(50, len(probs))
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            
            # 转换为字符/词，不过滤长度
            pred_dict = {}
            for prob, idx in zip(top_k_probs, top_k_indices):
                token = self.tokenizer.decode([idx]).strip()
                
                # 只过滤特殊标记和空白
                if token and token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                    # 移除可能的特殊字符标记（WordPiece）
                    token = token.replace('##', '')
                    if token and not token.isspace():
                        pred_dict[token] = float(prob)
            
            return pred_dict
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  BERT 预测失败: {e}")
            return {}
    
    def _score_grid_candidates(
        self,
        ocr_text: str,
        ocr_confidence: float,
        bert_predictions: Dict[str, float],
        ocr_candidates: List[Tuple[str, float]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        对格子的所有候选进行评分
        
        Args:
            ocr_text: OCR 识别的文本（如 '.co'）
            ocr_confidence: OCR 整体置信度
            bert_predictions: BERT 预测 {文本: 概率}
            ocr_candidates: OCR 的 Top-K 候选 [(文本, 概率), ...]
        
        Returns:
            [(候选文本, 融合分数, 详情), ...] 按分数降序排列
            详情包含: {'ocr_prob', 'bert_prob', 'source', 'match_type'}
        """
        candidates = []
        
        # 1. 评估 OCR Top-1
        ocr_bert_prob = bert_predictions.get(ocr_text, 0.0)
        ocr_score = self.fusion_weight * ocr_confidence + (1 - self.fusion_weight) * ocr_bert_prob
        candidates.append((
            ocr_text,
            ocr_score,
            {
                'ocr_prob': ocr_confidence,
                'bert_prob': ocr_bert_prob,
                'source': 'ocr_top1',
                'match_type': 'exact' if ocr_bert_prob > 0 else 'no_match'
            }
        ))
        
        # 2. 评估 OCR 其他候选（如果提供）
        if ocr_candidates:
            for ocr_cand_text, ocr_cand_prob in ocr_candidates[1:]:  # 跳过 Top-1
                bert_prob = bert_predictions.get(ocr_cand_text, 0.0)
                score = self.fusion_weight * ocr_cand_prob + (1 - self.fusion_weight) * bert_prob
                candidates.append((
                    ocr_cand_text,
                    score,
                    {
                        'ocr_prob': ocr_cand_prob,
                        'bert_prob': bert_prob,
                        'source': 'ocr_topk',
                        'match_type': 'exact' if bert_prob > 0 else 'no_match'
                    }
                ))
        
        # 3. 评估 BERT 独有的预测（不在 OCR 候选中的）
        ocr_texts = {ocr_text}
        if ocr_candidates:
            ocr_texts.update(c[0] for c in ocr_candidates)
        
        for bert_text, bert_prob in bert_predictions.items():
            if bert_text not in ocr_texts:
                # 对于 BERT 独有的候选，给 OCR 基础分
                # 检查部分匹配（例如 BERT 预测 "中国"，OCR 识别了 "中"）
                partial_match = any(
                    bert_text.startswith(ot) or ot.startswith(bert_text) 
                    for ot in ocr_texts if ot
                )
                
                if partial_match:
                    ocr_prob = 0.3  # 部分匹配给更高分
                    match_type = 'partial'
                else:
                    ocr_prob = 0.1  # 默认基础分
                    match_type = 'bert_only'
                
                score = self.fusion_weight * ocr_prob + (1 - self.fusion_weight) * bert_prob
                candidates.append((
                    bert_text,
                    score,
                    {
                        'ocr_prob': ocr_prob,
                        'bert_prob': bert_prob,
                        'source': 'bert_only',
                        'match_type': match_type
                    }
                ))
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def __repr__(self):
        return (f"GridContextEnhancer("
                f"model={self.model_name}, "
                f"context_window={self.context_window}, "
                f"fusion_weight={self.fusion_weight})")
