## 1. 环境准备
import sys
sys.path.append('..')

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# 配置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 导入自定义模块
from src.custom_ocr import CustomTextRecognizer, PostProcessingPipeline
from src.custom_ocr.processors import TopKDecoder, CTCDeduplicator, ConfidenceFilter

print("✓ 模块导入成功")
print("✓ Matplotlib 中文支持已配置")

## 2. 初始化识别器
# 初始化自定义识别器
recognizer = CustomTextRecognizer(
    model_name='PP-OCRv5_server_rec',
    device='gpu:0'  # 如果没有 GPU,改为 'cpu'
)

print("\n模型信息:")
model_info = recognizer.get_model_info()
for key, value in model_info.items():
    print(f"  {key}: {value}")
print("✓ OCR识别器初始化成功")

## 3. 加载测试数据 + 图像裁剪

# **【修改点】**: 从原图开始,调用裁剪函数生成temp_cropped.png

from pathlib import Path
import re
from PIL import Image

# 项目根目录
project_root = Path('..').resolve()

# ========== 【新增】图像裁剪步骤 ==========
print("【步骤0: 图像裁剪】")
print("="*80)

# 导入裁剪模块
from src.pipeline.image_cropper import crop_image, CROP_REGIONS

# 原始图像路径
raw_img_path = project_root / 'data' / 'samples' / '2022 第2題 (冬奧) (8份)_Original' / 'sample_04_01.png'
print(f"原始图像: {raw_img_path}")

# 判断使用哪个裁剪区域
filename = raw_img_path.name
match = re.search(r'_0(\d)\.png$', filename)

if match:
    page_num = int(match.group(1))
    print(f"检测到页码: {page_num}")
    
    if page_num in [1, 3]:
        crop_region = CROP_REGIONS['page1']
        region_name = 'page1'
    elif page_num in [2, 4]:
        crop_region = CROP_REGIONS['page2']
        region_name = 'page2'
    else:
        crop_region = None
        region_name = '无裁剪'
else:
    print("⚠️  无法从文件名提取页码,使用原图")
    crop_region = None
    region_name = '无裁剪'

print(f"使用裁剪区域: {region_name}")

# 执行裁剪或使用原图
if crop_region:
    cropped_img = crop_image(raw_img_path, crop_region)
    print(f"✓ 裁剪完成: {cropped_img.size}")
    print(f"  裁剪区域: {crop_region}")
else:
    cropped_img = Image.open(raw_img_path)
    print(f"✓ 使用原图: {cropped_img.size}")
    print(f"  无需裁剪")

# 保存图像
temp_cropped_path = project_root / 'output' / f'temp_cropped_{raw_img_path.stem}.png'
cropped_img.save(temp_cropped_path)
print(f"✓ 保存图像: {temp_cropped_path}")
print()

# ========== 【以下与02.8相同】加载图像和GT ==========
# 1. 加载图像
img_path = temp_cropped_path
img = cv2.imread(str(img_path))

if img is None:
    raise FileNotFoundError(f"无法加载图像: {img_path}\n请确保文件存在")

print(f"✓ 加载图像: {img.shape}")

# 2. 加载 Ground Truth(读取为字符串)
gt_file = project_root / 'data' / 'samples' / '2022 第2題 (冬奧) (8份)_Original' / 'sample_04_01_ground_truth.txt'
with open(gt_file, 'r', encoding='utf-8') as f:
    ground_truth = f.read()

print(f"✓ 加载 Ground Truth: {len(ground_truth)} 字符")

# 3. 清理 ground truth(移除 <insert></insert> 和 <?> 标记,移除换行和空格)
gt_clean = re.sub(r'<insert>|</insert>|<\?>', '', ground_truth)
gt_clean = gt_clean.replace('\n', '').replace(' ', '')

print(f"✓ 清理后的 Ground Truth: {len(gt_clean)} 字符")
print(f"前 100 字符: {gt_clean[:100]}")

## 3.1 格子检测与分割

# 从 src.preprocessing 导入格子检测方法
from src.preprocessing.grid_detection import detect_grid_lines, generate_grid_cells

# 转换为灰度图(detect_grid_lines 需要灰度图)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测格子线
print("检测格子线...")
lines = detect_grid_lines(image=gray, min_length=300)
horizontal_lines = lines['horizontal']
vertical_lines = lines['vertical']
print(f"✓ 检测到横线: {len(horizontal_lines)} 条")
print(f"✓ 检测到竖线: {len(vertical_lines)} 条")

# 生成格子
print("\n生成格子网格...")
cells = generate_grid_cells(horizontal_lines, vertical_lines, img.shape[:2])
print(f"✓ 生成格子: {len(cells)} 个")

# 过滤非空格子并裁剪图像(改进版:裁剪中心区域 + 提升阈值)
print("\n过滤非空格子并裁剪...")
cell_images = []
non_empty_cells = []

for cell in cells:
    x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    cell_img = img[y1:y2, x1:x2]
    
    # 判断是否为非空格子
    if cell_img.size > 0:
        gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        
        # 裁剪中心区域(避免边框干扰)
        h, w = gray_cell.shape
        crop = int(min(h, w) * 0.08)  # 裁剪 8%
        if h > 2*crop and w > 2*crop:  # 确保裁剪后还有内容
            center_region = gray_cell[crop:h-crop, crop:w-crop]
        else:
            center_region = gray_cell  # 格子太小,不裁剪
        
        # 计算中心区域的非白色像素比例
        non_white_ratio = (center_region < 240).sum() / center_region.size
        if non_white_ratio > 0.005:  # 超过0.5%的非白色像素认为是非空
            cell_images.append(cell_img)
            non_empty_cells.append(cell)

print(f"✓ 非空格子数: {len(cell_images)}")
print(f"✓ Ground Truth 字符数: {len(gt_clean)}")

## 4. 辅助函数定义

# 定义在整个 notebook 中复用的辅助函数
def restore_empty_cells(results_list, cells, non_empty_cells):
    """
    恢复空格子,构建完整的预测结果
    
    参数:
        results_list: OCR识别结果列表(非空格子)
        cells: 所有格子列表(包括空格子)
        non_empty_cells: 非空格子列表
        
    返回:
        prediction_data_full: 完整预测结果(包含空格子)
    """
    # 1. 创建完整的格子字典(包括空格子)
    all_cells_dict = {}
    for cell in cells:
        row, col = cell['row'], cell['col']
        all_cells_dict[(row, col)] = {
            'row': row,
            'col': col,
            'text': '',           # 默认为空
            'confidence': 1.0,    # 空格子置信度设为 1.0
            'is_empty': True      # 标记为空格子
        }
    
    # 2. 填充非空格子的识别结果
    for i, (result, cell) in enumerate(zip(results_list, non_empty_cells)):
        row, col = cell['row'], cell['col']
        all_cells_dict[(row, col)] = {
            'row': row,
            'col': col,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'is_empty': False,
            'ocr_result': result  # 保存完整的 OCR 结果(包含 top_k 等)
        }
    
    # 3. 转换为列表(按 row, col 排序)
    prediction_data_full = sorted(all_cells_dict.values(), 
                                  key=lambda x: (x['row'], x['col']))
    
    return prediction_data_full


def reconstruct_cell_data(metrics_dict):
    """
    从 by_row 结构重构完整的格子列表
    
    参数:
        metrics_dict: GridAccuracyCalculator 返回的结果字典
        
    返回:
        pred_cells: 带元数据的预测格子列表
        gt_cells: Ground Truth 文本列表
    """
    pred_cells = []
    gt_cells = []
    
    for row_result in metrics_dict['by_row']:
        row_idx = row_result['row']
        for col_idx, (pred_text, gt_text, conf) in enumerate(
            zip(row_result['pred_cells'], row_result['gt_cells'], 
                row_result['confidences'])
        ):
            pred_cells.append({
                'row': row_idx,
                'col': col_idx,
                'text': pred_text,
                'confidence': conf,
                'is_empty': (pred_text == '')
            })
            gt_cells.append({
                'row': row_idx,
                'col': col_idx,
                'text': gt_text
            })
    
    return pred_cells, gt_cells


def display_row_comparison(row_idx, pred_cells, gt_cells, enhanced_cells=None):
    """
    显示单行的对比(原始预测 vs Ground Truth vs 增强预测)
    
    参数:
        row_idx: 行索引
        pred_cells: 原始预测的格子列表
        gt_cells: Ground Truth 的格子列表
        enhanced_cells: 增强预测的格子列表(可选)
    """
    print(f"\n{'='*80}")
    print(f"第 {row_idx} 行对比")
    print(f"{'='*80}")
    
    # 获取该行的格子
    pred_row = [c for c in pred_cells if c['row'] == row_idx]
    gt_row = [c for c in gt_cells if c['row'] == row_idx]
    
    # 对齐到相同长度
    max_len = max(len(pred_row), len(gt_row))
    
    # 打印表头
    print(f"{'Col':<5} {'预测':<10} {'GT':<10}", end='')
    if enhanced_cells:
        enhanced_row = [c for c in enhanced_cells if c['row'] == row_idx]
        print(f" {'增强':<10} {'改变':<6}", end='')
    print(f" {'匹配':<6}")
    print(f"{'-'*80}")
    
    # 打印每列
    for col in range(max_len):
        pred_text = pred_row[col]['text'] if col < len(pred_row) else ''
        gt_text = gt_row[col]['text'] if col < len(gt_row) else ''
        
        # 判断是否匹配
        is_match = (pred_text == gt_text)
        match_str = '✓' if is_match else '✗'
        
        print(f"{col:<5} {pred_text:<10} {gt_text:<10}", end='')
        
        if enhanced_cells and col < len(enhanced_row):
            enhanced_text = enhanced_row[col]['text']
            is_changed = (enhanced_text != pred_text)
            changed_str = '→' if is_changed else ''
            enhanced_match = (enhanced_text == gt_text)
            enhanced_match_str = '✓' if enhanced_match else '✗'
            print(f" {enhanced_text:<10} {changed_str:<6}", end='')
            print(f" {match_str} → {enhanced_match_str}")
        else:
            print(f" {match_str}")
    

print(f"{'-'*80}")


print("✓ 辅助函数定义完成")

## 5. 全局处理器初始化

#创建共享的处理器实例,在整个 notebook 中复用(避免重复加载 BERT 模型)
# 导入处理器
from src.custom_ocr.processors.grid_context_enhancer import GridContextEnhancer

# 初始化共享的处理器实例
print("初始化全局处理器...")

# 1. Top-K 解码器
decoder = TopKDecoder(k=5)
print("✓ Top-K 解码器 (k=5)")

# 2. CTC 去重器
deduplicator = CTCDeduplicator()
print("✓ CTC 去重器")

# 3. 置信度过滤器
conf_filter = ConfidenceFilter(threshold=0.3)
print("✓ 置信度过滤器 (threshold=0.3)")

# 4. 格子级 BERT 增强器
grid_enhancer = GridContextEnhancer(
    model_name='bert-base-chinese',
    device='cuda:0',  # 如果没有 GPU,改为 'cpu'
    context_window=10,    # 上下文窗口:前后各5个格子
    confidence_threshold=0.8  # 低于此置信度的格子才会被BERT增强
)
print("✓ 格子级 BERT 增强器")
print(f"  - 模型: {grid_enhancer.model_name}")
print(f"  - 设备: {grid_enhancer.device}")
print(f"  - 窗口大小: {grid_enhancer.context_window}")
print(f"  - 置信度阈值: {grid_enhancer.confidence_threshold}")

print("\n所有处理器初始化完成!")

#---
# Part 2: 原始 OCR Pipeline(方法链式调用)
#---

## 6. 批量处理所有格子(方法链式调用)

# 使用**方法链式调用**(而非 Pipeline),每一步的输出都可见
# 批量处理所有格子
print("批量处理所有格子...")
print("="*80)

# 步骤 1: 获取原始概率矩阵(批量)
print("\n步骤 1: 获取原始概率矩阵...")
batch_raw_outputs = recognizer.batch_predict_with_raw_output(cell_images)
print(f"✓ 获取 {len(batch_raw_outputs)} 个格子的概率矩阵")

# 步骤 2: 方法链式调用(每步可见)
print("\n步骤 2: 执行后处理...")
results_list = []

for i, raw_output in enumerate(batch_raw_outputs):
    # 2.1 Top-K 解码
    decoded = decoder(raw_output)
    
    # 2.2 CTC 去重
    deduped = deduplicator(decoded)
    
    # 2.3 置信度过滤
    filtered = conf_filter(deduped)
    
    results_list.append(filtered)
    
    # 打印前10个格子的结果
    if i < 10:
        text = filtered.get('text', '')
        confidence = filtered.get('confidence', 0)
        print(f"  格子 {i+1:3d}: '{text:8s}' (置信度: {confidence:.3f})")

print(f"\n✓ 处理完成!共处理 {len(results_list)} 个非空格子")

# 步骤 3: 恢复空格子
print("\n步骤 3: 恢复空格子...")
prediction_data_full = restore_empty_cells(results_list, cells, non_empty_cells)

print(f"✓ 完整预测结果: {len(prediction_data_full)} 个格子")
print(f"  - 非空格子: {len([c for c in prediction_data_full if not c['is_empty']])} 个")
print(f"  - 空格子: {len([c for c in prediction_data_full if c['is_empty']])} 个")

# 验证:显示第一行的格子分布
first_row_cells = [c for c in prediction_data_full if c['row'] == 0]
print(f"\n第一行格子分布(共 {len(first_row_cells)} 个):")
for cell in first_row_cells[:10]:  # 只显示前10个
    status = "空" if cell['is_empty'] else f"'{cell['text']}'"
    print(f"  col {cell['col']}: {status}")

# ---
# Part 3: 格子级 BERT 增强
# ---

## 10. 批量 BERT 增强(格子级)

# 使用**格子级 BERT 增强器**批量处理所有非空格子

# 批量 BERT 增强(使用全局共享的 grid_enhancer)
print("批量 BERT 增强...")
print("="*80)

# 使用格子级 BERT 增强器
enhanced_results_list = grid_enhancer.enhance_grids(
    grid_results=results_list,
    grid_indices=None  # None 表示处理所有格子
)

print(f"✓ 增强完成!共处理 {len(enhanced_results_list)} 个格子")

# 统计增强效果
changed_count = 0
for original, enhanced in zip(results_list, enhanced_results_list):
    if original.get('text', '') != enhanced.get('text', ''):
        changed_count += 1

print(f"\n增强统计:")
print(f"  改变的格子数: {changed_count}")
print(f"  改变率: {changed_count/len(results_list):.2%}")

# 显示前10个改变的格子
print(f"\n前10个改变的格子:")
shown = 0
for i, (original, enhanced) in enumerate(zip(results_list, enhanced_results_list)):
    if original.get('text', '') != enhanced.get('text', ''):
        print(f"  格子 {i+1}: '{original.get('text', '')}' → '{enhanced.get('text', '')}' "
              f"(置信度: {original.get('confidence', 0):.3f} → "
              f"{enhanced.get('confidence', 0):.3f})")
        shown += 1
        if shown >= 10:
            break

# 恢复空格子
print(f"\n恢复空格子...")
enhanced_prediction_data_full = restore_empty_cells(
    enhanced_results_list, cells, non_empty_cells
)

print(f"✓ 完整增强预测结果: {len(enhanced_prediction_data_full)} 个格子")
print(f"  - 非空格子: {len([c for c in enhanced_prediction_data_full if not c['is_empty']])} 个")
print(f"  - 空格子: {len([c for c in enhanced_prediction_data_full if c['is_empty']])} 个")

## 12. 保存结果(纯文本格式)

# **【修改点】**: 保存简单的文本文件,而不是复杂的JSON格式

def extract_prediction_text_with_spacing(prediction_data_full):
    """
    从预测数据中提取文本,保留所有空格(包括行首、行中、行尾)
    
    参数:
        prediction_data_full: 完整的预测结果(包含空格子)
        
    返回:
        text: 格式化的文本字符串(保留所有空格)
    """
    # 按行分组
    rows_dict = {}
    for cell in prediction_data_full:
        row = cell['row']
        if row not in rows_dict:
            rows_dict[row] = []
        rows_dict[row].append(cell)
    
    # 按行号排序
    sorted_rows = sorted(rows_dict.items())
    
    # 构建文本
    lines = []
    for row_idx, row_cells in sorted_rows:
        # 按列号排序
        sorted_cells = sorted(row_cells, key=lambda x: x['col'])
        
        # 构建该行文本(空格子用空格' '表示)
        row_text = ''
        for cell in sorted_cells:
            if cell['is_empty']:
                row_text += ' '  # 空格子用空格表示
            else:
                row_text += cell['text']
        
        lines.append(row_text)
    
    return '\n'.join(lines)


def save_prediction_text(prediction_data_full, raw_img_path, mode='original'):
    """
    保存预测文本,保留目录结构
    
    参数:
        prediction_data_full: 完整的预测结果
        raw_img_path: 原始图像路径 (Path对象)
        mode: 'original' 或 'enhanced'
        
    返回:
        output_path: 保存的文件路径
    """
    # 提取文本
    pred_text = extract_prediction_text_with_spacing(prediction_data_full)
    
    # 构建输出路径
    project_root = Path('..').resolve()
    samples_dir = project_root / 'data' / 'samples'
    
    # 获取相对路径(相对于samples目录)
    try:
        relative_path = raw_img_path.relative_to(samples_dir)
    except ValueError:
        # 如果不在samples目录下,使用默认路径
        relative_path = Path('default') / raw_img_path.name
    
    # 构建输出目录(保留主题文件夹结构)
    output_base = project_root / 'output' / 'predictions'
    topic_folder = relative_path.parent  # 获取主题文件夹名
    output_dir = output_base / topic_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出文件名
    img_stem = raw_img_path.stem  # 例如 'sample_04_01'
    output_filename = f"{img_stem}_{mode}.txt"
    output_path = output_dir / output_filename
    
    # 保存文本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pred_text)
    
    return output_path


# 保存原始预测和增强预测
print("保存预测结果...")
print("="*80)

# 1. 保存原始预测
original_path = save_prediction_text(prediction_data_full, raw_img_path, mode='original')
print(f"✓ 原始预测已保存到: {original_path}")

# 2. 保存增强预测
enhanced_path = save_prediction_text(enhanced_prediction_data_full, raw_img_path, mode='enhanced')
print(f"✓ 增强预测已保存到: {enhanced_path}")

# 3. 显示保存的文件信息
print(f"\n文件信息:")
print(f"  原始预测: {original_path.stat().st_size} 字节")
print(f"  增强预测: {enhanced_path.stat().st_size} 字节")

# 4. 显示前5行预览
print(f"\n原始预测前5行预览:")
with open(original_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  行{i+1}: {repr(line.rstrip())}")  # 使用repr显示空格

print(f"\n增强预测前5行预览:")
with open(enhanced_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  行{i+1}: {repr(line.rstrip())}")

# 5. 总结
print(f"\n{'='*80}")
print("处理总结")
print(f"{'='*80}")
print(f"  原图 → 裁剪 → 格子检测 → OCR识别 → BERT增强")
print(f"\n  输出目录: {project_root / 'output' / 'predictions'}")
print(f"  - {original_path.name}")
print(f"  - {enhanced_path.name}")