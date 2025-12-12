"""
端到端OCR处理流程 - Pipeline V2

功能:
    1. 从原始图像裁剪作文区域
    2. 检测格子网格并分割单个格子
    3. 批量OCR识别(PaddleOCR + 后处理)
    4. BERT上下文增强
    5. 保存预测结果(纯文本格式)

作者: writtingOCR项目
日期: 2025-12-01
"""

# 标准库导入
import sys
import re
from pathlib import Path

# 第三方库导入
import numpy as np
import cv2
from PIL import Image

# 添加项目路径
sys.path.append('..')

# 自定义模块导入
from src.custom_ocr import CustomTextRecognizer
from src.custom_ocr.processors import TopKDecoder, CTCDeduplicator, ConfidenceFilter
from src.custom_ocr.processors.grid_context_enhancer import GridContextEnhancer
from src.preprocessing.grid_detection import detect_grid_lines, generate_grid_cells
from src.pipeline.image_cropper import crop_image, CROP_REGIONS


# ============================================================================
# 配置常量
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_DEVICE = 'gpu:0'
OCR_MODEL = 'PP-OCRv5_server_rec'

# OCR处理参数
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.3

# BERT增强参数
BERT_MODEL = 'bert-base-chinese'
BERT_DEVICE = 'cuda:0'
BERT_CONFIDENCE_THRESHOLD = 0.8
BERT_CONTEXT_WINDOW = 10

# 格子检测参数
GRID_MIN_LENGTH = 300
NON_WHITE_THRESHOLD = 0.005
CENTER_CROP_RATIO = 0.08


# ============================================================================
# 辅助函数定义
# ============================================================================

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
    # 创建完整的格子字典(包括空格子)
    all_cells_dict = {}
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
    
    # 转换为列表(按 row, col 排序)
    prediction_data_full = sorted(all_cells_dict.values(), 
                                  key=lambda x: (x['row'], x['col']))
    
    return prediction_data_full


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
        
        # 构建该行文本(空格子用空格表示)
        row_text = ''
        for cell in sorted_cells:
            if cell['is_empty']:
                row_text += ' '
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
    samples_dir = PROJECT_ROOT / 'data' / 'samples'
    
    # 获取相对路径(相对于samples目录)
    try:
        relative_path = raw_img_path.relative_to(samples_dir)
    except ValueError:
        # 如果不在samples目录下,使用默认路径
        relative_path = Path('default') / raw_img_path.name
    
    # 构建输出目录(保留主题文件夹结构)
    output_base = PROJECT_ROOT / 'output' / 'predictions'
    topic_folder = relative_path.parent
    output_dir = output_base / topic_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出文件名
    img_stem = raw_img_path.stem
    output_filename = f"{img_stem}_{mode}.txt"
    output_path = output_dir / output_filename
    
    # 保存文本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pred_text)
    
    return output_path


# ============================================================================
# 主流程开始
# ============================================================================

print("="*80)
print("端到端OCR处理流程启动")
print("="*80)

# 初始化OCR识别器
print("\n[1/6] 初始化OCR识别器...")
recognizer = CustomTextRecognizer(
    model_name=OCR_MODEL,
    device=DEFAULT_DEVICE
)
print(f"✓ OCR模型加载成功: {OCR_MODEL}")


# 图像路径配置
raw_img_path = PROJECT_ROOT / 'data' / 'samples' / '2022 第2題 (冬奧) (8份)_Original' / 'sample_04_01.png'
gt_file = PROJECT_ROOT / 'data' / 'samples' / '2022 第2題 (冬奧) (8份)_Original' / 'sample_04_01_ground_truth.txt'

print(f"\n[2/6] 图像预处理与裁剪...")
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

# 保存裁剪后的图像
temp_cropped_path = PROJECT_ROOT / 'output' / f'temp_cropped_{raw_img_path.stem}.png'
cropped_img.save(temp_cropped_path)
print(f"✓ 裁剪图像已保存: {temp_cropped_path.name}")

# 加载裁剪后的图像
img = cv2.imread(str(temp_cropped_path))
if img is None:
    raise FileNotFoundError(f"无法加载图像: {temp_cropped_path}")
print(f"✓ 图像加载成功: {img.shape}")

print(f"\n[3/6] 格子检测与分割...")

# 转换为灰度图
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

print(f"\n[4/6] OCR识别...")

# 初始化处理器
from src.custom_ocr.processors.grid_context_enhancer import GridContextEnhancer
print("初始化BERT增强器...")

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

print(f"\n✓ 处理完成!共 {len(results_list)} 个非空格子")

# 恢复空格子
prediction_data_full = restore_empty_cells(results_list, cells, non_empty_cells)
print(f"✓ 完整结果: {len(prediction_data_full)} 个格子 ({len([c for c in prediction_data_full if not c['is_empty']])} 非空)")

print(f"\n[5/6] BERT增强...")

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

print(f"✓ 增强完成: {changed_count} 个格子改变 ({changed_count/len(results_list):.1%})")

# 恢复空格子
enhanced_prediction_data_full = restore_empty_cells(
    enhanced_results_list, cells, non_empty_cells
)

print(f"\n[6/6] 保存结果...")

# 保存结果
original_path = save_prediction_text(prediction_data_full, raw_img_path, mode='original')
enhanced_path = save_prediction_text(enhanced_prediction_data_full, raw_img_path, mode='enhanced')

print(f"✓ 原始预测: {original_path.name}")
print(f"✓ 增强预测: {enhanced_path.name}")

print(f"\n{'='*80}")
print(f"处理完成!")
print(f"  原始图像: {raw_img_path.name}")
print(f"  输出目录: {PROJECT_ROOT / 'output' / 'predictions'}")

# 5. 总结
print(f"\n{'='*80}")
print("处理总结")
print(f"{'='*80}")
print(f"  原图 → 裁剪 → 格子检测 → OCR识别 → BERT增强")
print(f"\n  输出目录: {PROJECT_ROOT / 'output' / 'predictions'}")
print(f"  - {original_path.name}")
print(f"  - {enhanced_path.name}")