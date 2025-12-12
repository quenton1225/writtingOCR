"""
ContextEnhancer - BERT 上下文增强处理器

利用预训练的 BERT 模型进行完形填空，结合 OCR 概率和语言模型概率，
对识别结果进行上下文感知的修正。

Author: OCR Pipeline Team
Date: 2025-11-26
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import torch
    from transformers import BertTokenizer, BertForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  警告: transformers 未安装，ContextEnhancer 将无法使用")
    print("   安装命令: pip install transformers torch")


class ContextEnhancer:
    """
    上下文增强处理器
    
    使用 BERT 完形填空模型，结合上下文信息对 OCR 识别结果进行修正。
    
    工作原理:
    1. 对每个字符位置，提取前后 N 个字符作为上下文
    2. 将目标字符替换为 [MASK]，输入 BERT 模型
    3. BERT 预测该位置最可能的字符
    4. 融合 OCR 概率和 BERT 概率: P_final = α × P_ocr + (1-α) × P_bert
    5. 根据融合后的概率重新选择最佳字符
    
    参数:
        model_name: BERT 模型名称，默认 'bert-base-chinese'
        context_window: 上下文窗口大小（前后各取 N 个字符），默认 5
        fusion_weight: OCR 概率权重（0-1），默认 0.7 (70% OCR + 30% BERT)
        confidence_threshold: 仅修正低于该置信度的字符，默认 0.5
        correction_threshold: 修正阈值，融合概率需比原概率高出此值才修正，默认 0.1
        max_candidates: 考虑的候选字符数量，默认 5
        device: 计算设备 ('cpu', 'cuda', 'cuda:0' 等)，默认自动检测
        batch_size: BERT 推理批大小，默认 32
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-chinese',
        context_window: int = 5,
        fusion_weight: float = 0.7,
        confidence_threshold: float = 0.5,
        correction_threshold: float = 0.1,
        max_candidates: int = 5,
        device: Optional[str] = None,
        batch_size: int = 32
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
        
        # 保存配置
        self.model_name = model_name
        self.context_window = context_window
        self.fusion_weight = fusion_weight
        self.confidence_threshold = confidence_threshold
        self.correction_threshold = correction_threshold
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        
        # 设备配置
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 延迟加载模型（首次使用时加载）
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
        
        print(f"✓ ContextEnhancer 初始化完成")
        print(f"  模型: {model_name}")
        print(f"  上下文窗口: ±{context_window} 字符")
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
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline 调用接口
        
        Args:
            data: 输入数据字典
            
        Returns:
            处理后的数据字典
        """
        return self.process(data)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行上下文增强处理
        
        Args:
            data: 包含以下字段的字典:
                - decoded_text: List[str] - 已解码的文本
                - top_k_chars: List[List[str]] - Top-K 候选字符
                - top_k_probs: List[List[float]] - 对应概率
                - character_list: List[str] - 字符表
        
        Returns:
            增强后的数据字典，添加以下字段:
                - context_enhanced_text: List[str] - BERT 增强后的文本
                - bert_corrections: List[List[Dict]] - 修正记录
                - fusion_scores: List[List[float]] - 融合后的概率
        """
        # 延迟加载模型
        self._load_model()
        
        # 提取输入数据（批格式但只有1个元素）
        decoded_texts = data.get('decoded_text', [])
        top_k_chars = data.get('top_k_chars', [])
        top_k_probs = data.get('top_k_probs', [])
        character_list = data.get('character_list', [])
        
        if len(decoded_texts) == 0:
            print("⚠️  警告: decoded_text 为空，跳过上下文增强")
            return data
        
        # 提取第一个元素（实际只处理1个样本）
        text = decoded_texts[0]
        candidates = top_k_chars[0] if len(top_k_chars) > 0 else []
        probs = top_k_probs[0] if len(top_k_probs) > 0 else []
        
        # 对单个文本进行增强
        enhanced_text, corrections, fusion_scores = self._enhance_single_text(
            text, candidates, probs, character_list
        )
        
        # 更新数据（保持批格式约定）
        data['text'] = enhanced_text
        data['bert_corrections'] = [corrections]  # 保持列表格式
        data['fusion_scores'] = [fusion_scores]
        
        # 统计修正信息
        print(f"✓ 上下文增强完成: 修正 {len(corrections)} 个字符")
        
        return data
    
    def _enhance_single_text(
        self,
        text: str,
        candidates: List[List[str]],
        probs: List[List[float]],
        character_list: List[str]
    ) -> Tuple[str, List[Dict], List[float]]:
        """
        对单个文本进行上下文增强
        
        Args:
            text: 原始文本
            candidates: 每个位置的候选字符列表
            probs: 每个位置的候选概率
            character_list: 完整字符表
        
        Returns:
            (enhanced_text, corrections, fusion_scores)
        """
        if not text:
            return text, [], []
        
        # 准备批量 BERT 输入
        masked_inputs = []
        positions_to_check = []
        
        for i in range(len(text)):
            # 获取该位置的 OCR 置信度
            ocr_confidence = probs[i][0] if i < len(probs) and len(probs[i]) > 0 else 1.0
            
            # 只对低置信度字符进行 BERT 增强
            if ocr_confidence < self.confidence_threshold:
                masked_input = self._create_masked_input(text, i)
                masked_inputs.append(masked_input)
                positions_to_check.append(i)
                print(f"[调试] 位置 {i} (字符='{text[i]}', 置信度={ocr_confidence:.3f})")
                print(f"       BERT 输入: '{masked_input}'")
        
        # 如果没有需要检查的位置，直接返回
        if not positions_to_check:
            print("[调试] 所有字符置信度都足够高，无需 BERT 增强")
            return text, [], [1.0] * len(text)
        
        print(f"\n[调试] 将检查 {len(positions_to_check)} 个低置信度字符")
        
        # 批量获取 BERT 预测
        bert_predictions = self._batch_predict_bert(masked_inputs)
        
        # 逐位置融合概率并决定是否修正
        enhanced_chars = list(text)
        corrections = []
        fusion_scores = [1.0] * len(text)
        
        for idx, pos in enumerate(positions_to_check):
            bert_pred = bert_predictions[idx]
            
            print(f"\n[调试] 位置 {pos} BERT 预测 Top-5:")
            if bert_pred:
                sorted_pred = sorted(bert_pred.items(), key=lambda x: x[1], reverse=True)[:5]
                for char, prob in sorted_pred:
                    print(f"       '{char}': {prob:.4f}")
            else:
                print(f"       (无有效预测)")
            
            # 融合 OCR 和 BERT 概率
            fused_char, fused_prob, should_correct = self._fuse_and_decide(
                pos, text[pos], candidates, probs, bert_pred, character_list
            )
            
            print(f"       融合结果: '{fused_char}' (prob={fused_prob:.4f}, 修正={should_correct})")
            
            fusion_scores[pos] = fused_prob
            
            # 如果需要修正
            if should_correct and fused_char != text[pos]:
                enhanced_chars[pos] = fused_char
                corrections.append({
                    'position': pos,
                    'original': text[pos],
                    'corrected': fused_char,
                    'ocr_prob': probs[pos][0] if pos < len(probs) and len(probs[pos]) > 0 else 0.0,
                    'bert_prob': bert_pred.get(fused_char, 0.0),
                    'fused_prob': fused_prob
                })
        
        enhanced_text = ''.join(enhanced_chars)
        return enhanced_text, corrections, fusion_scores
    
    def _create_masked_input(self, text: str, position: int) -> str:
        """
        创建带 [MASK] 的输入文本
        
        Args:
            text: 完整文本
            position: 要 mask 的位置
        
        Returns:
            带 [MASK] 的文本
        """
        # 提取左上下文
        left_start = max(0, position - self.context_window)
        left_context = text[left_start:position]
        
        # 提取右上下文
        right_end = min(len(text), position + self.context_window + 1)
        right_context = text[position + 1:right_end]
        
        # 组合
        masked_text = left_context + '[MASK]' + right_context
        return masked_text
    
    def _batch_predict_bert(self, masked_inputs: List[str]) -> List[Dict[str, float]]:
        """
        批量进行 BERT 预测
        
        Args:
            masked_inputs: 包含 [MASK] 的输入文本列表
        
        Returns:
            每个输入的 Top-K 预测结果 [{char: prob, ...}, ...]
        """
        all_predictions = []
        
        # 分批处理
        for i in range(0, len(masked_inputs), self.batch_size):
            batch = masked_inputs[i:i + self.batch_size]
            batch_predictions = self._predict_bert_batch(batch)
            all_predictions.extend(batch_predictions)
        
        return all_predictions
    
    def _predict_bert_batch(self, batch_texts: List[str]) -> List[Dict[str, float]]:
        """
        对一个批次进行 BERT 预测
        
        Args:
            batch_texts: 批量文本
        
        Returns:
            预测结果列表
        """
        # Tokenize
        inputs = self.tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # 解析结果
        results = []
        for batch_idx in range(len(batch_texts)):
            # 找到 [MASK] 的位置
            mask_token_index = (inputs.input_ids[batch_idx] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_token_index) == 0:
                results.append({})
                continue
            
            mask_token_index = mask_token_index[0].item()
            
            # 获取 [MASK] 位置的预测概率
            mask_token_logits = predictions[batch_idx, mask_token_index]
            probs = torch.softmax(mask_token_logits, dim=0)
            
            # 获取 Top-K
            top_k_probs, top_k_indices = torch.topk(probs, min(self.max_candidates, len(probs)))
            
            # 转换为字符-概率字典
            pred_dict = {}
            for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
                token = self.tokenizer.decode([idx])
                # 过滤特殊 token 和多字符 token
                if token and len(token) == 1 and token not in ['[', ']', '#']:
                    pred_dict[token] = float(prob)
            
            results.append(pred_dict)
        
        return results
    
    def _fuse_and_decide(
        self,
        position: int,
        original_char: str,
        candidates: List[List[str]],
        probs: List[List[float]],
        bert_pred: Dict[str, float],
        character_list: List[str]
    ) -> Tuple[str, float, bool]:
        """
        融合 OCR 和 BERT 概率，决定是否修正
        
        Args:
            position: 字符位置
            original_char: 原始字符
            candidates: OCR 候选列表
            probs: OCR 概率列表
            bert_pred: BERT 预测结果 {char: prob}
            character_list: 字符表
        
        Returns:
            (最佳字符, 融合概率, 是否应该修正)
        """
        # 获取该位置的 OCR 候选
        ocr_candidates = candidates[position] if position < len(candidates) else []
        ocr_probs = probs[position] if position < len(probs) else []
        
        if not ocr_candidates or not bert_pred:
            return original_char, 1.0, False
        
        # 构建融合概率字典
        fused_probs = {}
        
        # 考虑所有出现在 OCR 或 BERT 中的候选
        all_candidates = set(ocr_candidates[:self.max_candidates]) | set(bert_pred.keys())
        
        for char in all_candidates:
            # OCR 概率
            ocr_prob = 0.0
            if char in ocr_candidates:
                idx = ocr_candidates.index(char)
                if idx < len(ocr_probs):
                    ocr_prob = ocr_probs[idx]
            
            # BERT 概率
            bert_prob = bert_pred.get(char, 0.0)
            
            # 融合
            fused_prob = self.fusion_weight * ocr_prob + (1 - self.fusion_weight) * bert_prob
            fused_probs[char] = fused_prob
        
        # 找出最佳候选
        best_char = max(fused_probs, key=fused_probs.get)
        best_prob = fused_probs[best_char]
        
        # 原始字符的融合概率
        original_prob = fused_probs.get(original_char, 0.0)
        
        # 决定是否修正: 最佳候选的融合概率需显著高于原始字符
        should_correct = (
            best_char != original_char and
            best_prob > original_prob + self.correction_threshold
        )
        
        return best_char, best_prob, should_correct
    
    def visualize(self, data: Dict[str, Any]) -> str:
        """
        可视化上下文增强结果
        
        Args:
            data: 处理后的数据
        
        Returns:
            可视化字符串
        """
        lines = []
        lines.append("=" * 80)
        lines.append("上下文增强可视化")
        lines.append("=" * 80)
        
        corrections = data.get('bert_corrections', [])
        
        if not corrections or not corrections[0]:
            lines.append("未进行任何修正")
            return '\n'.join(lines)
        
        # 显示每个修正
        for batch_idx, batch_corrections in enumerate(corrections):
            if not batch_corrections:
                continue
            
            lines.append(f"\n文本 {batch_idx + 1}: 共 {len(batch_corrections)} 处修正")
            lines.append("-" * 80)
            
            for i, corr in enumerate(batch_corrections):
                lines.append(
                    f"{i+1:2d}. 位置 {corr['position']:3d}: "
                    f"{corr['original']} → {corr['corrected']} | "
                    f"OCR: {corr['ocr_prob']:.3f}, "
                    f"BERT: {corr['bert_prob']:.3f}, "
                    f"融合: {corr['fused_prob']:.3f}"
                )
        
        lines.append("=" * 80)
        return '\n'.join(lines)
    
    def get_config(self) -> Dict[str, Any]:
        """返回处理器配置"""
        return {
            'name': 'ContextEnhancer',
            'model_name': self.model_name,
            'context_window': self.context_window,
            'fusion_weight': self.fusion_weight,
            'confidence_threshold': self.confidence_threshold,
            'correction_threshold': self.correction_threshold,
            'max_candidates': self.max_candidates,
            'device': self.device,
            'batch_size': self.batch_size
        }
