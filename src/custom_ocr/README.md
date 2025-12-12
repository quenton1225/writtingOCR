# 自定义 OCR 后处理框架使用指南

## 📦 已创建文件

```
src/custom_ocr/
├── __init__.py                   # 包入口
├── recognizer.py                 # 自定义识别器（获取概率矩阵）
├── pipeline.py                   # 后处理管道框架
└── processors/
    ├── __init__.py               # 处理器包
    ├── topk_decoder.py           # Top-K 解码器
    ├── ctc_deduplicator.py       # CTC 去重器
    └── confidence_filter.py      # 置信度过滤器

notebooks/
└── 02.7_custom_postprocessing.ipynb  # 测试 notebook
```

## 🚀 快速开始

### 1. 基础使用

```python
from src.custom_ocr import CustomTextRecognizer, PostProcessingPipeline
from src.custom_ocr.processors import TopKDecoder, CTCDeduplicator, ConfidenceFilter

# 初始化识别器
recognizer = CustomTextRecognizer(model_name='PP-OCRv5_server_rec')

# 创建处理管道
pipeline = PostProcessingPipeline(recognizer.get_character_list())
pipeline.add_processor(TopKDecoder(k=5)) \
        .add_processor(CTCDeduplicator()) \
        .add_processor(ConfidenceFilter(threshold=0.3))

# 识别图像
raw_output = recognizer.predict_with_raw_output('image.jpg')
result = pipeline.process(raw_output)

# 查看结果
print(f"识别文本: {result['text']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 2. 获取 Top-K 候选

```python
# 查看 Top-5 候选字符
for t in range(min(5, len(result['top_k_chars'][0]))):
    chars = result['top_k_chars'][0][t]
    probs = result['top_k_probs'][0][t]
    print(f"时间步 {t}: {list(zip(chars, probs))}")
```

### 3. 识别低置信度字符

```python
# 获取需要增强的字符位置
low_conf_positions = result['low_confidence_positions'][0]
text = result['text']

print(f"需要增强的字符: {len(low_conf_positions)} 个")
for pos in low_conf_positions:
    print(f"  位置 {pos}: '{text[pos]}'")
```

### 4. 批量处理

```python
# 处理多个格子
results = []
for cell_img in cell_images:
    raw_output = recognizer.predict_with_raw_output(cell_img)
    result = pipeline.process(raw_output)
    results.append(result['text'])

# 拼接结果
full_text = ''.join(results)
```

### 5. 切换不同策略

```python
# 保守策略（高精度）
conservative = PostProcessingPipeline(recognizer.get_character_list())
conservative.add_processor(TopKDecoder(k=3)) \
            .add_processor(CTCDeduplicator()) \
            .add_processor(ConfidenceFilter(threshold=0.5))

# 激进策略（高召回）
aggressive = PostProcessingPipeline(recognizer.get_character_list())
aggressive.add_processor(TopKDecoder(k=10)) \
          .add_processor(CTCDeduplicator()) \
          .add_processor(ConfidenceFilter(threshold=0.2))
```

## 📊 关键数据结构

### CustomTextRecognizer.predict_with_raw_output() 返回

```python
{
    'prob_matrix': np.ndarray,      # [batch, time_steps, num_classes]
    'raw_image': np.ndarray,        # 原始图像
    'character_list': list,         # 字符映射表
    'batch_size': int,
    'time_steps': int,
    'num_classes': int
}
```

### Pipeline.process() 返回

```python
{
    # 原始数据
    'prob_matrix': np.ndarray,
    'character_list': list,
    
    # TopKDecoder 添加
    'top_k_indices': np.ndarray,    # [batch, time_steps, k]
    'top_k_probs': np.ndarray,      # [batch, time_steps, k]
    'top_k_chars': list,            # [batch][time_step][k]
    
    # CTCDeduplicator 添加
    'decoded_text': list,           # 解码后的文本
    'text': str,                    # 单样本简化输出
    'decoded_indices': list,        # 去重后的索引
    'char_positions': list,         # 字符位置
    'avg_confidence': list,
    'confidence': float,            # 单样本简化输出
    
    # ConfidenceFilter 添加
    'low_confidence_positions': list,  # 低置信度位置
    'confidence_flags': list,          # 每个字符的标志
    'low_confidence_ratio': float,     # 低置信度比例
    
    # Pipeline 元数据
    'pipeline_log': list            # 执行日志
}
```

## 🔧 处理器参数

### TopKDecoder
- `k` (int): Top-K 数量，默认 5
- `return_scores` (bool): 是否归一化概率，默认 True

### CTCDeduplicator
- `blank_idx` (int): Blank 标记索引，默认 0
- `mode` (str): 去重模式，默认 'standard'

### ConfidenceFilter
- `threshold` (float): 置信度阈值，默认 0.3
- `strategy` (str): 过滤策略，'flag' | 'mark' | 'remove'，默认 'flag'

## 🐛 调试功能

### 可视化 CTC 对齐

```python
ctc = CTCDeduplicator()
# ... 执行 pipeline ...
print(ctc.visualize_ctc_alignment(result, sample_idx=0))
```

### 可视化置信度

```python
conf_filter = ConfidenceFilter(threshold=0.3)
# ... 执行 pipeline ...
print(conf_filter.visualize_confidence(result, sample_idx=0))
```

### 查看执行日志

```python
result = pipeline.process(data)
for log in result['pipeline_log']:
    print(f"步骤 {log['step']}: {log['processor']} - {log['status']}")
```

## 📝 下一步

1. **运行 02.7 notebook** 进行基础测试
2. **从 02.6 加载格子数据** 进行批量测试
3. **对比准确率** 与原方案比较
4. **实现 ContextEnhancer** 用于上下文增强

## ⚠️ 注意事项

1. 首次运行会下载模型（约 500MB）
2. 需要安装 PaddleOCR：`pip install paddleocr`
3. GPU 推荐，CPU 也可运行（较慢）
4. 概率矩阵较大，注意内存使用
