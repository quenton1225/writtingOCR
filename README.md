# 手写作文自动评分系统

一个基于OCR和自然语言处理的中文手写作文自动评分系统。

## 🎯 项目愿景

**长期目标**: 输入学生作文扫描件 → 输出评分区间/具体分数

该系统将实现:
- 📝 自动识别手写文字内容
- ✏️ 理解学生的修改符号(删除、插入等)
- 🤖 基于语言模型理解作文质量
- 📊 结合评分规则给出合理分数

## 🗺️ 整体架构

```
[作文扫描件] 
    ↓
阶段一: OCR文字提取 (当前开发中)
  ├─ 手写文字识别
  ├─ 删除符号检测
  ├─ 插入符号识别  
  └─ 文本重建
    ↓
阶段二: 语言理解与评分 (未来)
  ├─ 语言模型分析(内容、结构、语言)
  ├─ 基于规则的检查(错别字、语法等)
  └─ 综合评分模型
    ↓
[评分结果输出]
```

## 📍 当前进度

- 🚧 **开发阶段**: 阶段一 - OCR文字提取
- 📅 **当前任务**: Stage 0 环境搭建(已完成) → Stage 1 基准OCR测试
- 📊 **数据规模**: 220份学生作文(2021-2025年)

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境(推荐)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

**注意**: 
- `pdf2image` 需要安装 [poppler](https://github.com/oschwartz10612/poppler-windows/releases/) (Windows)
- PaddleOCR首次运行会自动下载模型文件

### 2. 提取测试样本

```bash
python scripts/extract_samples.py
```

这会从 `data/raw/Original/` 提取少量作文到 `data/samples/` 用于测试。

### 3. 运行基准OCR测试

```bash
# 待完成 - Stage 1
jupyter notebook notebooks/01_baseline_ocr_test.ipynb
```

## 项目结构

```
writtingOCR/
├── data/
│   ├── raw/                    # 原始PDF数据
│   │   ├── Original/           # 学生原始作文
│   │   └── Marked/             # 老师批改版本(后期使用)
│   ├── samples/                # 测试样本
│   ├── processed/              # 预处理后的数据
│   └── results/                # 识别结果
├── src/
│   ├── utils/                  # 工具函数
│   │   └── pdf_to_images.py   # PDF转图片
│   ├── ocr/                    # OCR模块
│   ├── symbol_detection/       # 符号检测模块
│   ├── semantic/               # 语义理解模块
│   ├── visualization/          # 可视化模块
│   └── pipeline.py             # 主流程
├── notebooks/                  # 实验notebooks
├── scripts/                    # 脚本工具
├── tests/                      # 单元测试
├── docs/                       # 文档
├── PROJECT_PLAN.md            # 项目计划
├── ROADMAP.md                 # 详细路线图
└── requirements.txt           # Python依赖
```

## 📚 文档

- [ROADMAP.md](ROADMAP.md) - **阶段一**详细开发路线图(OCR部分,6个Stages)
- [PROJECT_PLAN.md](PROJECT_PLAN.md) - 完整项目计划

## 🚀 阶段一开发进度 (OCR文字提取)

**目标**: 从作文扫描件中准确提取文字和修改符号

- [x] **Stage 0**: 环境准备 ✅
  - 项目结构、PDF转图片工具、依赖配置
- [ ] **Stage 1**: 基准OCR测试 (下一步)
  - 评估PaddleOCR在手写作文上的表现
- [ ] **Stage 2**: 删除符号检测
  - 识别学生的删除线标记
- [ ] **Stage 3**: 符号-文字关联
  - 确定哪些文字被删除/插入
- [ ] **Stage 4**: 完整Pipeline
  - 端到端处理流程
- [ ] **Stage 5**: 全量测试优化
  - 在220份作文上验证

详细技术路线见 [ROADMAP.md](ROADMAP.md)

## 🔮 阶段二规划 (语言理解与评分)

**未来工作** (阶段一完成后启动):

1. **语言模型集成**
   - 使用LLM分析作文内容、结构、语言质量
   - 参考评分标准文档(`data/raw/寫作評分參照標準 2025.09.02.pdf`)

2. **规则系统开发**
   - 错别字检测
   - 语法检查
   - 字数统计
   - 格式规范

3. **评分模型训练**
   - 利用Marked数据(老师批注)作为训练标签
   - 开发可解释的评分模型
   - 给出分数及理由

4. **系统集成**
   - 整合OCR和评分模块
   - 开发用户界面
   - 批量处理能力

## 🎓 项目特点

- **渐进式开发**: 从小样本验证到全量部署
- **真实数据**: 220份跨5年的真实学生作文
- **两阶段架构**: OCR提取 + 智能评分
- **可解释性**: 不仅给分数,还要说明理由

## License

MIT