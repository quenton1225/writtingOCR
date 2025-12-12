# 项目路线图 - 手写作文删除符号识别

## 🎯 最终目标

**输入**: 学生手写作文图片(包含删除线等修改符号)  
**输出**: 文本 + 删除标记位置

```
示例输出:
"你<del>再</del>在哪？"
或
"你[删除]在哪？"
```

**核心要求**:
- ✅ 识别删除痕迹的位置
- ✅ 识别周围未删除的文字
- ❌ 不需要识别被删除的具体文字内容

---

## 📋 阶段拆分

### Stage 0: 环境准备 (1-2天)

#### 任务清单
- [ ] 创建完整项目目录结构
- [ ] 安装Python依赖
  - PyMuPDF / pdf2image (PDF处理)
  - OpenCV (图像处理)
  - PaddleOCR (文字识别)
  - Matplotlib (可视化)
- [ ] 编写PDF转图片脚本
- [ ] 提取测试样本(5-10篇作文)

#### 输出
- `src/utils/pdf_to_images.py` - PDF转图片工具
- `data/samples/` - 测试样本图片

#### 验收标准
- 成功从PDF提取单页图片
- 每2页合并为1份作文(如需要)
- 图片质量足够清晰(300 DPI以上)

---

### Stage 1: 纯文字识别基准 (2-3天)

#### 目标
忽略所有符号,只识别文字内容

#### 任务清单
- [ ] 集成PaddleOCR
- [ ] 对测试样本进行文字识别
- [ ] 评估识别准确率
- [ ] 记录问题案例:
  - 哪些字容易识别错?
  - 格子线是否影响识别?
  - 删除线区域的识别情况如何?

#### 输出
- `src/ocr/text_recognizer.py` - OCR封装模块
- `notebooks/01_baseline_ocr_test.ipynb` - 测试notebook
- `docs/stage1_baseline_report.md` - 基准测试报告

#### 评估指标
```python
{
    "total_chars": 1234,
    "correct_chars": 1180,
    "accuracy": 0.956,
    "common_errors": ["的/地混淆", "相似字错误"],
    "deletion_zone_accuracy": 0.85  # 删除线区域准确率
}
```

#### 验收标准
- 整体文字识别准确率 > 85%
- 理解格子线对识别的影响
- 确认删除线区域的识别现状

---

### Stage 2: 删除线检测 (3-5天)

#### 目标
识别图片中所有的删除线位置

#### 方法探索
两种并行方案:

**方案A: 基于规则的检测**
```python
流程:
1. 灰度化、二值化
2. 霍夫直线检测
3. 筛选特征:
   - 短线(长度 < 3个字符宽度)
   - 与文字区域重叠
   - 横向或斜向(排除格子线)
4. 输出删除线bbox
```

**方案B: 基于深度学习的检测**
```python
流程:
1. 标注20-30个删除线样本
2. 训练YOLOv8-nano检测器
3. 类别: [deletion_mark]
4. 输出删除线bbox
```

#### 任务清单
- [ ] 实现规则检测原型
- [ ] 在10份样本上测试效果
- [ ] 记录召回率和误检率
- [ ] 如果规则方法召回率 < 80%, 启动深度学习方案

#### 输出
- `src/symbol_detection/deletion_detector.py` - 删除线检测模块
- `notebooks/02_deletion_detection.ipynb` - 检测实验
- `data/visualizations/` - 检测结果可视化

#### 评估指标
```python
{
    "deletion_lines_detected": 15,
    "deletion_lines_gt": 18,
    "recall": 0.833,
    "precision": 0.882,
    "false_positives": 2  # 误检的格子线或下划线
}
```

#### 验收标准
- 删除线召回率 > 75%
- 误检率 < 10%
- 能区分删除线 vs 格子线 vs 好句子下划线

---

### Stage 3: 删除线与文字关联 (2-3天)

#### 目标
确定删除线覆盖了哪些文字

#### 技术方案
```python
输入:
  - OCR结果: [{"text": "在", "bbox": [x1,y1,x2,y2]}]
  - 删除线: [{"bbox": [x1,y1,x2,y2]}]

算法:
  for each 删除线:
    affected_chars = []
    for each 文字:
      if IoU(删除线, 文字) > 0.3:  # 重叠超过30%
        affected_chars.append(文字)
    
    输出: {
      "deletion": 删除线位置,
      "deleted_text": affected_chars
    }
```

#### 任务清单
- [ ] 实现IoU(交并比)计算
- [ ] 实现文字-删除线匹配算法
- [ ] 处理边界情况:
  - 删除线跨多个字
  - 删除线只覆盖半个字
  - 多条删除线重叠
- [ ] 生成结构化输出

#### 输出
- `src/semantic/text_deletion_linker.py` - 关联模块
- `notebooks/03_text_deletion_linking.ipynb` - 测试notebook

#### 输出格式
```json
{
  "original_text": "你再在哪？",
  "text_with_deletions": [
    {"text": "你", "deleted": false},
    {"text": "再", "deleted": true},
    {"text": "在", "deleted": false},
    {"text": "哪", "deleted": false},
    {"text": "？", "deleted": false}
  ],
  "readable_output": "你<del>再</del>在哪？"
}
```

#### 验收标准
- 准确关联删除线与文字
- 输出格式清晰可读
- 处理边界情况不崩溃

---

### Stage 4: 完整Pipeline集成 (2-3天)

#### 目标
端到端系统,从图片到标注文本

#### 系统架构
```python
class WritingOCRPipeline:
    def process(self, image_path):
        # 1. OCR识别
        ocr_result = self.text_recognizer.recognize(image)
        
        # 2. 删除线检测
        deletions = self.deletion_detector.detect(image)
        
        # 3. 关联分析
        linked_result = self.linker.link(ocr_result, deletions)
        
        # 4. 格式化输出
        return {
            "plain_text": "你在哪？",           # 清洁版本
            "with_deletions": "你<del>再</del>在哪？",  # 带标记
            "structured": linked_result        # 结构化数据
        }
```

#### 任务清单
- [ ] 实现主Pipeline类
- [ ] 批量处理接口
- [ ] 添加日志和错误处理
- [ ] 命令行工具
- [ ] 可视化结果(在原图上标注)

#### 输出
- `src/pipeline.py` - 主流程
- `scripts/batch_process.py` - 批量处理脚本
- `src/visualization/annotator.py` - 结果可视化

#### 验收标准
- 可以处理单张图片
- 可以批量处理整个文件夹
- 输出多种格式(JSON, TXT, HTML)
- 生成可视化对比图

---

### Stage 5: 全量测试与优化 (1-2周)

#### 目标
在220份作文上验证系统性能

#### 测试策略
```
训练集/测试集划分:
- 2021-2023数据: 训练/调优 (98份)
- 2024数据: 验证集 (25份)
- 2025数据: 测试集 (97份)
```

#### 任务清单
- [ ] 处理所有220份作文
- [ ] 人工抽查50份结果
- [ ] 统计错误类型:
  - 删除线漏检
  - 删除线误检
  - 文字识别错误
  - 关联错误
- [ ] 针对性优化
- [ ] 生成性能报告

#### 输出
- `docs/full_evaluation_report.md` - 完整评估报告
- `data/results/` - 所有作文的识别结果

#### 性能目标
| 指标 | 目标值 |
|------|--------|
| 文字识别准确率 | > 90% |
| 删除线召回率 | > 85% |
| 删除线精确率 | > 90% |
| 端到端准确率 | > 80% |

---

### Stage 6: 扩展功能 (可选)

#### 其他符号类型
- [ ] 插入符号识别(^, ∧)
- [ ] 圈注符号识别
- [ ] 批注文字识别

#### 高级功能
- [ ] Web界面(Streamlit)
- [ ] 实时预览
- [ ] 结果导出(Word, PDF)

---

## 🔄 迭代策略

### 快速原型 → 逐步完善
```
Week 1: Stage 0-1 完成
  ↓ 验证文字识别可行性
Week 2-3: Stage 2-3 完成
  ↓ 验证删除线检测可行性
Week 4: Stage 4 完成
  ↓ 端到端验证
Week 5-6: Stage 5 完成
  ↓ 全量测试和优化
```

### 决策点
| 阶段 | 决策问题 | 判断标准 | 行动 |
|------|---------|---------|------|
| Stage 1 | 格子线是否影响OCR? | 准确率 < 80% | 添加格子线去除预处理 |
| Stage 2 | 规则方法是否足够? | 召回率 < 75% | 启动深度学习方案 |
| Stage 3 | 关联算法是否准确? | 错误率 > 15% | 调整IoU阈值或算法 |

---

## 📊 里程碑

| 里程碑 | 完成标志 | 预计时间 |
|--------|---------|----------|
| M0 | 提取10份测试样本 | Day 2 |
| M1 | 基准OCR报告完成 | Week 1 |
| M2 | 删除线检测原型可用 | Week 3 |
| M3 | 端到端Pipeline运行 | Week 4 |
| M4 | 50份作文测试通过 | Week 5 |
| M5 | 全量220份处理完成 | Week 6 |

---

## 🎓 学习与调整

### 每个阶段结束后:
1. **回顾**: 什么工作得好?什么不好?
2. **记录**: 更新文档,记录问题和解决方案
3. **调整**: 根据结果调整后续阶段计划

### 关键问题记录
- [ ] 删除线的典型长度/角度/粗细
- [ ] 学生常用的删除方式(单线/双线/涂黑)
- [ ] 格子纸规格(每格尺寸)
- [ ] 扫描质量参数

---

## 🚀 立即开始

### 下一步行动(本周):
1. ✅ 创建项目结构
2. ✅ 安装依赖
3. ⏩ 编写PDF转图片脚本
4. ⏩ 提取2-3份测试样本
5. ⏩ 运行第一次OCR测试

---

**文档版本**: v1.0  
**最后更新**: 2025-01-21  
**维护者**: Quenton
