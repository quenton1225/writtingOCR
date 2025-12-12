# 手写作文批改符号识别系统 - 项目计划

## 🎯 当前项目状态 (2025-12-04)

**开发阶段**: 阶段二完成,进入批量化生产阶段  
**核心成果**: 端到端OCR Pipeline V3 已可用  
**准确率**: 原始OCR 70-72%, BERT增强后提升至 75-80%  
**处理能力**: 支持批量处理,已成功处理 21 份作文 (42 张图像)  
**数据规模**: 2022-2023年共 21 份学生作文样本  
**最新里程碑**: Pipeline V3 批量处理系统 (2025-12-01)

---

## 项目概述

### 整体目标
开发分阶段的手写作文识别与评分系统:

**阶段一 (当前)**: 手写文字高精度识别 ✅ **基本完成**
- 文字内容识别 (PaddleOCR + BERT增强)
- 批量处理能力
- 准确率: 75-80%

**阶段二 (规划)**: 批改符号检测与关联
- 删除线、插入符号检测
- 符号与文字的关联分析
- 状态: 待评估需求后启动

**阶段三 (未来)**: 自动评分系统
- 基于语言模型的内容分析
- 结合评分规则的综合评估
- 状态: 概念阶段

### 技术栈
- **语言**: Python 3.10
- **核心库**: 
  - PaddleOCR (PP-OCRv5_server_rec) - 文字识别
  - Transformers + BERT (bert-base-chinese) - 上下文增强
  - OpenCV - 图像处理与格子检测
  - PyTorch - 深度学习后端
- **可选**: YOLOv8 (符号检测,未使用)

---

## 阶段一: 基础验证与环境搭建 ✅ 已完成

### 1.1 环境准备 ✅

- [x] Python 3.10 环境 + 虚拟环境配置
- [x] 核心依赖安装:
  - ✅ PaddleOCR (PP-OCRv5_server_rec)
  - ✅ OpenCV, NumPy, PIL
  - ✅ Transformers (BERT)
  - ✅ PyTorch (BERT后端)

**关键配置**:
- NumPy 版本: 1.26.4 (兼容性要求)
- 虚拟环境路径: `venv/`
- 必须激活虚拟环境运行 (详见 pipeline_v3.py 文档)

### 1.2 数据收集 ✅

- [x] 样本数据收集完成:
  - **2022年冬奥主题**: 8份作文 (16张图像)
  - **2023年艺术节主题**: 13份作文 (26张图像)
  - **总计**: 21份作文, 42张图像
- [x] 样本存储: `data/samples/`
  - `2022 第2題 (冬奧) (8份)_Original/`
  - `2023 第2題 (藝術節) (13份)_Original/`
- [x] Ground Truth 标注完成 (每个样本对应 `*_ground_truth.txt`)

### 1.3 基准测试 ✅ 决策已做出

**测试目标**: 确定是否需要去除格子线

**实验方案** (详见 `01_baseline_ocr_test.ipynb`, `MEMO.md`):
- 实验组A: 全图直接OCR
- 实验组B: 裁剪作文区域后OCR
- 实验组C: 不同PaddleOCR参数配置

**测试结论**:
- ❌ 全图OCR准确率: ~30-40% (包含页眉、页脚、表格等噪声)
- ✅ 裁剪后准确率: ~70-72% (仅作文正文区域)
- ✅ 格子线不影响识别 (无需去除)
- ✅ 最佳参数: PaddleOCR 默认配置即可

**关键发现**:
1. **必须进行图像裁剪**: 页面布局噪声严重影响识别
2. **两种页面格式**: 
   - 第1页 (_01, _03): `CROP_REGION['page1']`
   - 第2页 (_02, _04): `CROP_REGION['page2']`
3. **格子线处理**: 不需要去除,PaddleOCR 可直接识别

**技术决策**:
- ✅ 实现自动裁剪模块 (`src/pipeline/image_cropper.py`)
- ✅ 根据文件名自动判断页面类型
- ❌ 不开发格子线去除功能

---

## 阶段二: 文字识别模块 ✅ 已完成

## 阶段二: 文字识别模块 ✅ 已完成

### 2.1 图像预处理管线 ✅

**已实现模块**:

- [x] **自动裁剪** (`src/pipeline/image_cropper.py`)
  - 支持两种页面格式 (page1, page2)
  - 根据文件名自动判断 (\_01/\_03 → page1, \_02/\_04 → page2)
  - CROP_REGIONS 配置化管理

- [x] **格子检测与分割** (`src/preprocessing/grid_detection.py`)
  - `detect_grid_lines()`: 霍夫变换检测横竖线
  - `generate_grid_cells()`: 生成格子坐标矩阵
  - 非空格子过滤 (中心区域像素比例检测)
  - 阈值: `NON_WHITE_THRESHOLD = 0.005` (0.5%)

**未实现模块** (验证后不需要):
- ❌ 图像校正 (透视变换、旋转) - 样本质量良好
- ❌ 格子线去除 - 不影响识别
- ❌ 复杂图像增强 - 默认参数已足够

### 2.2 文字识别 ✅ 超预期完成

**核心模块**: `src/custom_ocr/`

- [x] **CustomTextRecognizer** (`recognizer.py`)
  - PaddleOCR 封装与优化
  - 批量处理支持 (`batch_predict_with_raw_output`)
  - 返回原始概率矩阵用于后处理

- [x] **后处理 Pipeline** (`processors/`)
  - **TopKDecoder**: Top-K 候选解码 (k=5)
  - **CTCDeduplicator**: CTC 序列去重
  - **ConfidenceFilter**: 置信度过滤 (threshold=0.3)
  - 方法链式调用: `decoder → deduplicator → filter`

- [x] **格子级批量处理**
  - 支持数百个格子并行识别
  - 空格子自动恢复 (`restore_empty_cells`)
  - 行列结构化输出

**输出格式**:
```python
{
  "text": "你",
  "confidence": 0.95,
  "top_k": [("你", 0.95), ("仍", 0.03), ("你", 0.01)],
  "is_empty": False,
  "row": 0,
  "col": 3
}
```

### 2.3 文字后处理 ✅ 创新性增强

**🌟 核心创新: BERT 上下文增强**

- [x] **GridContextEnhancer** (`processors/grid_context_enhancer.py`)
  - 利用 BERT 完形填空能力修正低置信度字符
  - 格子级上下文建模 (前后10格窗口)
  - OCR + BERT 概率融合: `P_final = 0.6 × P_ocr + 0.4 × P_bert`
  - Delta 回退机制: 只接受明显改进 (threshold=0.15)

**技术原理**:
1. 提取所有格子文本构建上下文序列
2. 将低置信度格子替换为 [MASK] 标记
3. BERT 预测 [MASK] 位置应该是什么字符
4. 融合 OCR 和 BERT 概率分布
5. 仅当提升显著时才修正

**实验验证** (`notebooks/02.8_grid_bert_enhancement.ipynb`):
- **参数优化**: 网格搜索 (详见 `scripts/grid_search_bert_params.py`)
  - 最佳 fusion_weight: 0.6
  - 最佳 delta_threshold: 0.15
  - 搜索空间: 108 种配置组合
- **效果统计**:
  - 改变率: ~5-8% 格子被修正
  - 准确率提升: +3-5% (相对提升)
  - 误修正率: <1%

**性能指标**:
- 推理时间: ~2-3秒/图 (BERT 部分)
- 内存占用: ~2GB (BERT 模型)
- 批量处理: 支持 (共享模型加载)

### 2.4 评估系统 ✅

- [x] **GridAccuracyCalculator** (`src/evaluation/grid_accuracy.py`)
  - 字符级准确率计算
  - 行级准确率分析
  - 与 Ground Truth 自动对比
  - 混淆矩阵统计

**准确率演进**:
```
基准 (全图): ~35%
↓ 裁剪预处理
第一次突破: ~71% (adaptive_cropping_results.json)
↓ 格子检测+自定义处理
第二次突破: ~70.65% (accuracy_comparison_final.json)
↓ BERT上下文增强
最终水平: ~75-80% (基于部分样本评估)
```

---

## 阶段二.5: 端到端 Pipeline 集成 ✅ 已完成

### 2.5.1 Pipeline 演进历程

**V1 - 原型验证** (已删除):
- 单文件处理流程验证
- 手动指定输入输出路径
- 无批量处理能力

**V2 - 模块重构** (`scripts/pipeline_v2.py`):
- 代码重构: 580 → 427 行
- 引入 `src/` 模块化架构
- 改进错误处理和日志
- 仍为单文件处理

**V3 - 生产批量化** (`scripts/pipeline_v3.py`) ✅ **当前版本**:
- **核心功能**:
  - 批量文件夹扫描 (`scan_input_folder`)
  - 自动过滤非原始图像 (`_marked`, `_visualization`, etc.)
  - 自动页面判断 (\_01/\_03 → page1, \_02/\_04 → page2)
  - 保留目录结构输出 (predictions/)
  
- **错误恢复**:
  - 单图错误不影响批处理
  - 详细错误日志记录
  - 进度追踪和统计报告

- **命令行接口**:
  ```bash
  python scripts/pipeline_v3.py data/samples --output output/predictions
  ```

- **代码规模**: 483 行 (功能更强但更简洁)
- **环境要求**: ⚠️ 必须在虚拟环境中运行 (NumPy 1.26.4)

### 2.5.2 批量处理能力

**已处理数据**:
- 21 篇作文 (42 张图像)
- 2022 冬奥主题: 8 篇
- 2023 艺术节主题: 13 篇

**处理速度**:
- 单图处理: ~30-45 秒
- 批量吞吐: ~2-3 分钟/篇作文
- 瓶颈: BERT 推理 (~2-3秒/图)

**输出结构**:
```
output/predictions/
  └── 2022 第2題 (冬奧) (8份)_Original/
      ├── sample_001_res.json
      ├── sample_002_res.json
      └── ...
  └── 2023 第2題 (藝術節) (13份)_Original/
      └── ...
```

---

## 阶段三: 符号识别模块 📝 规划调整中

**原计划**: 实现删除线、插入符等编辑符号的自动识别

**当前状态**: **暂缓/重新评估**

**调整原因**:
1. **数据不足**: 当前 21 篇作文中符号样本稀少
2. **优先级**: 文字识别准确率提升更重要
3. **投入产出比**: 需要大量标注数据但影响有限

**下一步决策点**:
- [ ] 收集更多包含符号的样本数据
- [ ] 评估人工标注符号的可行性
- [ ] 考虑跳过符号识别,直接进入评分系统验证

**技术储备** (设计保留供将来参考):

### 3.1 符号分类定义
定义需要识别的符号类型:

| 符号类型 | 描述 | 优先级 |
|---------|------|--------|
| 删除线 | 横线穿过文字 | P0 |
| 插入符号 | ^, ∧ 在字间/行间 | P0 |
| 调换符号 | 弧线或数字标注 | P1 |
| 圈注符号 | 圈出错误部分 | P1 |
| 波浪线 | 强调标记 | P2 |
| 其他 | 待补充 | P2 |

### 3.2 规则检测方法(快速原型)
**目标: 覆盖80%常见场景**

#### 删除线检测
```python
# 伪代码
1. 霍夫直线检测
2. 筛选水平短线
3. 判断与文字bbox的重叠
4. 排除格子线(长度、位置规律性)
```

#### 插入符号检测
```python
# 伪代码
1. 轮廓检测
2. 形状匹配(V形或^形)
3. 位置过滤(字间/行间)
4. 尺寸过滤(小于文字尺寸)
```

#### 其他符号
- 圈注: 闭合曲线 + 包含文字区域
- 调换: 曲线检测或小数字检测

### 3.3 深度学习检测器(可选增强)
**如果规则方法效果不足**

#### 数据标注
- [ ] 使用LabelImg/CVAT标注工具
- [ ] 标注符号边界框和类别
- [ ] 最小数据集: 100-200张图像

#### 模型训练
- [ ] 选择轻量级检测器(YOLOv8-nano 或 YOLOv5s)
- [ ] 训练配置:
  - 输入尺寸: 640x640
  - 数据增强: 旋转、缩放、亮度调整
  - 迁移学习: 使用预训练权重
- [ ] 评估指标: mAP, Precision, Recall

#### SAM 3实验(探索性)
- [ ] 测试SAM 3的符号分割能力
- [ ] 使用文本prompt: "deletion mark", "insertion symbol"
- [ ] 对比与YOLO的效果差异

---

## 阶段四: 实验与优化历程 📊

本阶段记录了从初始验证到生产化的完整实验过程,共产生 19 个 Jupyter Notebooks。

### 4.1 基础验证阶段 (Notebooks 01-02)

**01_baseline_ocr_test.ipynb**: 初始 OCR 基准测试
- 直接对整图应用 PaddleOCR
- 结果: 准确率 ~35%, 格子结构混乱
- 结论: 必须添加预处理步骤

**02_cropped_ocr_test.ipynb**: 裁剪预处理测试
- 实现自动裁剪,去除页面边缘
- 多参数组合测试 (detection_threshold, unclip_ratio)
- 突破: 准确率提升至 ~71%
- 输出: `output/cropped_ocr_results/` (6 个测试组)

### 4.2 格子检测与处理 (Notebooks 02.5-02.7)

**02.5_vertical_line_detection_test.ipynb**: 纵向线条检测
- 霍夫变换检测格子线
- 垂直线提取和过滤

**02.6_grid_based_ocr.ipynb**: 格子级 OCR
- 格子分割算法实现
- 逐格子 OCR 识别
- 结果: 准确率 ~70.65%

**02.7_custom_postprocessing.ipynb**: 自定义后处理
- 实现 TopKDecoder, CTCDeduplicator
- 置信度过滤策略

**02.71_grid_detection_debug.ipynb**: 格子检测调试
- 可视化格子检测结果
- 调优检测参数

### 4.3 BERT 增强突破 (Notebooks 02.8-02.9x)

**02.8_grid_bert_enhancement.ipynb**: 🌟 核心突破
- 首次引入 BERT 上下文增强
- 格子级完形填空策略
- 融合权重和 delta 阈值设计
- 效果: 准确率提升至 ~75-80%

**02.9_trocr_validation.ipynb**: TrOCR 对比验证
- 测试 Microsoft TrOCR 模型
- 对比 PaddleOCR vs TrOCR
- 结论: PaddleOCR + BERT 更适合本场景

**02.91_trocr_validation.ipynb**: TrOCR 进一步验证

**02.92_easyocr_validation.ipynb**: EasyOCR 对比测试
- 测试 EasyOCR 引擎
- 多引擎横向对比

### 4.4 参数优化与对比 (Notebooks 03.x)

**03.0_pre_postprocessing_optimization.ipynb**: 前后处理优化
- 系统性测试不同参数组合
- 建立评估基准

**03.1_evaluation_system_enhancement.ipynb**: 评估系统增强
- GridAccuracyCalculator 实现
- 字符级和行级准确率计算

**03.2_preprocessing_comparison.ipynb**: 预处理方案对比
**03.3_preprocessing_comparison.ipynb**: 预处理方案对比 (更新版)

### 4.5 端到端 Pipeline 开发 (Notebooks 04.x)

**04.0_end_to_end_pipeline.ipynb**: 端到端原型
**04.1_end_to_end_pipeline.ipynb**: Pipeline 优化版
- 完整流程串联
- 为 scripts/pipeline_v1.py 奠定基础

### 4.6 支持实验产出

**脚本工具**:
- `scripts/grid_search_bert_params.py`: BERT 参数网格搜索
- `scripts/grid_search_delta_threshold.py`: Delta 阈值搜索
- `scripts/extract_samples.py`: 样本提取工具

**网格搜索结果**: `output/grid_search/`
- 108 种参数组合测试
- 最佳配置: fusion_weight=0.6, delta_threshold=0.15
- CSV 详细结果和文本摘要

---

## 阶段五: 项目结构与技术栈 📁

### 5.1 当前项目结构

### 4.1 语义理解模块
将检测到的符号与文字关联:

#### 删除线处理
```python
# 输出示例
{
  "type": "deletion",
  "affected_text": "的地得",
  "position": [x, y, w, h],
  "confidence": 0.92
}
```

#### 插入符号处理
```python
{
  "type": "insertion",
  "position_between": ["学校", "里"],
  "insert_point": [x, y]
}
```

#### 调换符号处理
```python
{
  "type": "swap",
  "swap_items": ["item1", "item2"],
  "order": [2, 1]
}
```

```

### 5.1 当前项目结构

```
writtingOCR/
├── data/                          # 数据目录
│   ├── samples/                   # 样本数据
│   │   ├── 2022 第2題 (冬奧) (8份)_Original/
│   │   └── 2023 第2題 (藝術節) (13份)_Original/
│   ├── raw/                       # 原始数据
│   │   ├── Original/
│   │   └── Marked/
│   ├── processed/                 # 处理后数据
│   ├── results/                   # 结果数据
│   └── visualizations/            # 可视化输出
│
├── src/                           # 源代码模块
│   ├── custom_ocr/                # 自定义OCR模块
│   │   ├── recognizer.py          # CustomTextRecognizer
│   │   └── processors/            # 后处理器
│   │       ├── grid_context_enhancer.py  # BERT增强
│   │       ├── topk_decoder.py
│   │       ├── ctc_deduplicator.py
│   │       └── confidence_filter.py
│   ├── preprocessing/             # 预处理模块
│   │   ├── grid_detection.py      # 格子检测
│   │   └── image_cropper.py       # 图像裁剪
│   ├── pipeline/                  # 流程模块
│   │   └── image_cropper.py       # Pipeline专用裁剪
│   ├── evaluation/                # 评估模块
│   │   └── grid_accuracy.py       # GridAccuracyCalculator
│   ├── utils/                     # 工具函数
│   ├── visualization/             # 可视化工具
│   └── semantic/                  # 语义处理 (未完成)
│
├── scripts/                       # 脚本工具
│   ├── pipeline_v3.py             # ✅ 生产批处理脚本
│   ├── pipeline_v2.py             # 重构版
│   ├── extract_samples.py         # 样本提取
│   ├── grid_search_bert_params.py # BERT参数搜索
│   └── grid_search_delta_threshold.py
│
├── notebooks/                     # Jupyter实验笔记本 (19个)
│   ├── 01_baseline_ocr_test.ipynb
│   ├── 02.8_grid_bert_enhancement.ipynb  # 🌟 BERT突破
│   ├── 04.0_end_to_end_pipeline.ipynb
│   └── ...
│
├── output/                        # 输出结果
│   ├── predictions/               # Pipeline输出
│   ├── grid_search/               # 参数搜索结果
│   ├── cropped_ocr_results/       # 裁剪测试结果
│   └── ...
│
├── tests/                         # 单元测试 (未完善)
├── docs/                          # 文档
├── venv/                          # 虚拟环境 ⚠️ 必需
├── requirements.txt               # 依赖清单
├── README.md                      # 项目说明
├── PROJECT_PLAN.md                # 本文件
├── ROADMAP.md                     # 路线图
└── MEMO.md                        # 开发笔记 (896行)
```

### 5.2 技术栈详细版本

**核心依赖**:
```
Python 3.10.x
NumPy 1.26.4          # ⚠️ 不能用2.x, pandas兼容性
PaddlePaddle 3.0.0    # PaddleOCR依赖
PaddleOCR 2.9.1       # PP-OCRv5引擎
transformers 4.46.3   # BERT模型
torch 2.5.1           # PyTorch后端
opencv-python 4.10.0  # 图像处理
pandas 2.2.3          # 数据处理
```

**模型文件**:
- PaddleOCR: PP-OCRv5_server_rec (自动下载)
- BERT: bert-base-chinese (HuggingFace)

### 5.3 环境配置要求

**必需步骤**:
1. 创建虚拟环境: `python -m venv venv`
2. 激活环境: `.\venv\Scripts\Activate.ps1` (Windows PowerShell)
3. 安装依赖: `pip install -r requirements.txt`
4. 验证 NumPy: `python -c "import numpy; print(numpy.__version__)"` → 应为 1.26.4

**PowerShell 执行策略** (Windows):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 阶段六: 下一步计划与里程碑 🎯

### 6.1 近期计划 (1-2周)

**P0 - 符号识别数据收集决策**:
- [ ] 评估是否有更多包含符号的作文样本
- [ ] 决定: 继续符号识别 vs 跳过进入评分系统
- [ ] 如继续: 制定数据标注计划

**P1 - 系统稳定性提升**:
- [ ] 完善单元测试覆盖 (`tests/`)
- [ ] 添加集成测试 (完整 Pipeline)
- [ ] 错误处理增强 (更多边界情况)

**P2 - 文档完善**:
- [x] 更新 PROJECT_PLAN.md (本次更新)
- [ ] 补充 API 文档 (src/ 模块)
- [ ] 添加使用示例和教程

### 6.2 中期目标 (1-2月)

**选项 A: 继续符号识别**:
- 收集更多样本 (目标: 50-100 篇含符号作文)
- 实现规则检测原型
- 评估准确率和实用性

**选项 B: 跳过符号, 构建评分系统**:
- 设计评分标准 (字数、错别字、修改次数等)
- 实现自动评分算法
- 与人工评分对比验证

### 6.3 长期愿景 (3-6月)

- [ ] 部署为 Web 服务 (Flask/FastAPI)
- [ ] 开发前端界面 (上传→处理→结果展示)
- [ ] 支持实时反馈和人工修正
- [ ] 建立完整的数据标注和训练循环

---

## 阶段七: 技术债与已知问题 ⚠️

### 7.1 环境问题

**NumPy 兼容性** (已解决但需注意):
- 问题: NumPy 2.x 与 pandas/PaddleOCR 不兼容
- 解决: 锁定 NumPy 1.26.4
- 风险: 未来依赖升级可能再次冲突
- 监控: 定期检查依赖兼容性

**虚拟环境依赖** (已文档化):
- 问题: 系统 Python 无法正常运行
- 解决: pipeline_v3.py 添加警告文档
- 改进: 考虑添加启动脚本自动检测环境

### 7.2 数据质量问题

**Ground Truth 格式不一致**:
- 部分样本: 逐格字符标注
- 部分样本: 整行文本标注
- 影响: 评估结果对比困难
- 待办: 统一 GT 格式规范

**样本量不足**:
- 当前: 21 篇作文 (42 图)
- 理想: 100+ 篇用于统计显著性
- 符号样本: 几乎没有
- 计划: 持续收集新数据

### 7.3 代码质量

**测试覆盖不足**:
- 单元测试: 几乎没有
- 集成测试: 手动验证为主
- CI/CD: 未建立
- 风险: 重构易引入 Bug

**文档不完整**:
- 代码注释: 部分模块缺失
- API 文档: 未生成
- 使用指南: README 简略
- 改进: 逐步补充 docstring

**硬编码配置**:
- 部分参数: 直接写在代码中 (如 CROP_REGIONS)
- 理想: 统一配置文件管理
- 待办: 考虑使用 YAML/JSON 配置

### 7.4 性能瓶颈

**BERT 推理慢**:
- 耗时: ~2-3秒/图
- 原因: CPU 推理, 模型较大
- 优化方向: GPU 加速, 模型量化

**批量处理顺序执行**:
- 当前: 逐图串行处理
- 改进: 多进程/异步并行
- 预期提升: 2-3x

---

## 阶段八: 成功经验总结 ✅

### 8.1 技术决策

**✅ 选择裁剪而非去格子线**:
- 验证: 格子线不影响识别
- 收益: 节省开发时间, 避免图像失真
- 准确率: 71% (裁剪) vs ~35% (全图)

**✅ BERT 上下文增强策略**:
- 创新: 格子级完形填空
- 效果: +3-5% 准确率提升
- 代价: +2-3秒/图 (可接受)
- 关键: Delta 阈值避免误修正

**✅ 参数网格搜索**:
- 方法: 系统化搜索而非手动调参
- 工具: `scripts/grid_search_bert_params.py`
- 发现: fusion_weight=0.6, delta_threshold=0.15 最优
- 收益: 数据驱动决策, 可重现

### 8.2 开发流程

**✅ Notebook 快速原型验证**:
- 策略: 新想法先在 Notebook 测试
- 优势: 可视化调试, 快速迭代
- 转化: 验证成功后迁移到 src/
- 记录: 保留完整实验历史

**✅ 渐进式 Pipeline 演进**:
- V1: 快速验证可行性
- V2: 重构提高可维护性
- V3: 批量化生产就绪
- 教训: 不要过早优化, 验证后再重构

**✅ 详细记录 MEMO.md**:
- 内容: 896 行开发笔记
- 价值: 决策记录, 问题追溯, 知识沉淀
- 建议: 每次重要进展及时记录

### 8.3 避坑指南

**❌ 不要忽视虚拟环境**:
- 教训: NumPy 2.x 兼容性问题
- 最佳实践: 始终在 venv 中开发和运行

**❌ 不要盲目追求复杂方案**:
- 案例: 去格子线实际不需要
- 原则: 先测试简单方案, 必要时再优化

**❌ 不要依赖单一样本评估**:
- 问题: 过拟合特定样本
- 解决: 多样本交叉验证, 数据集划分

---

## 附录: 关键文件索引 📚

### 核心代码

| 文件路径 | 功能 | 关键内容 |
|---------|------|---------|
| `scripts/pipeline_v3.py` | 生产批处理 | scan_input_folder, process_single_image |
| `src/custom_ocr/recognizer.py` | OCR封装 | CustomTextRecognizer |
| `src/custom_ocr/processors/grid_context_enhancer.py` | BERT增强 | GridContextEnhancer, delta机制 |
| `src/preprocessing/grid_detection.py` | 格子检测 | detect_grid_lines, generate_grid_cells |
| `src/evaluation/grid_accuracy.py` | 评估系统 | GridAccuracyCalculator |

### 关键实验

| Notebook | 突破点 | 输出结果 |
|---------|--------|---------|
| `02_cropped_ocr_test.ipynb` | 裁剪预处理 | 71% 准确率 |
| `02.8_grid_bert_enhancement.ipynb` | BERT增强 | 75-80% 准确率 |
| `03.1_evaluation_system_enhancement.ipynb` | 评估系统 | GridAccuracyCalculator |

### 配置与文档

| 文件 | 用途 |
|-----|------|
| `requirements.txt` | 依赖清单 |
| `MEMO.md` | 开发笔记 (896行) |
| `ROADMAP.md` | 技术路线图 |
| `README.md` | 项目说明 |

---

## 版本信息 📋

**当前版本**: v0.3.0 (Pipeline V3 生产化)

**更新历史**:
- 2025-12-04: 大规模更新 PROJECT_PLAN.md, 反映实际进展
- 2025-11-29: BERT 参数网格搜索完成
- 2025-11-28: Pipeline V3 批量处理上线
- 2025-11-XX: BERT 增强突破, 准确率 75-80%
- 2025-10-XX: 格子检测和裁剪预处理, 准确率 71%
- 2025-09-XX: 项目启动, 基准测试

**下次更新计划**: 符号识别决策后或评分系统开发时

---

## 项目状态概览 📊

| 指标 | 当前值 | 目标值 |
|-----|--------|--------|
| 文字识别准确率 | 75-80% | 85-90% |
| 处理速度 | ~30-45秒/图 | <20秒/图 |
| 已处理样本数 | 21篇作文 | 100+篇 |
| 代码测试覆盖 | <10% | 70%+ |
| 符号识别能力 | 未实现 | 待定 |
| 生产化就绪 | ✅ V3 Pipeline | ✅ |

**项目健康度**: 🟢 健康 (核心功能完成, 正常迭代)

**风险等级**: 🟡 中等 (数据量不足, 符号识别待定)

### 5.3 鲁棒性测试
- [ ] 边界case测试
  - 低质量扫描件
  - 深色格子线
  - 密集批改
- [ ] 错误分析与修正

### 5.4 用户界面(可选)
- [ ] 命令行工具
---

**文档版本**: v2.0  
**最后更新**: 2025-12-04  
**维护者**: Quenton

---

## 附注: 如何维护本文档

**何时更新**:
- ✅ 完成重要技术突破 (如 BERT 增强)
- ✅ 代码架构重大调整 (如 V2 → V3)
- ✅ 项目方向变化 (如符号识别暂缓)
- ✅ 每月例行检查更新状态

**更新方式**:
1. 修改对应阶段的状态标记 (❌/⏳/✅)
2. 补充实际完成情况和数据
3. 更新版本号和日期
4. 同步更新 MEMO.md 和 ROADMAP.md

**文档原则**:
- 保持事实准确, 不夸大成果
- 记录失败和教训, 不只记录成功
- 定期清理过时内容
- 保留技术决策过程供未来参考
