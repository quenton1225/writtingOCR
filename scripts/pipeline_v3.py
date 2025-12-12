"""
端到端OCR批量处理流程 - Pipeline V3

⚠️  重要: 必须在虚拟环境中运行
    
    1. 激活虚拟环境:
       Windows PowerShell:  .\venv\Scripts\Activate.ps1
       Windows CMD:         .\venv\Scripts\activate.bat
       Linux/Mac:           source venv/bin/activate
    
    2. 验证环境 (可选):
       python -c "import sys; print(sys.executable)"
       # 输出应包含: ...\\writtingOCR\\venv\\Scripts\\python.exe

功能:
    1. 批量扫描输入文件夹中的所有图像
    2. 自动裁剪作文区域(根据文件名判断页码)
    3. 检测格子网格并分割单个格子
    4. 批量OCR识别(PaddleOCR + 后处理)
    5. BERT上下文增强
    6. 保存预测结果(纯文本格式,保留目录结构)
    7. 生成处理报告

使用方法:
    # 处理默认文件夹 (data/samples)
    python scripts/pipeline_v3.py
    
    # 处理指定文件夹
    python scripts/pipeline_v3.py --input "data/samples/2022 第2題 (冬奧) (8份)_Original"
    
    # 查看帮助
    python scripts/pipeline_v3.py --help

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
# 批量处理函数
# ============================================================================

def scan_input_folder(folder_path):
    """
    扫描输入文件夹,获取所有需要处理的图像
    
    参数:
        folder_path: Path对象,输入文件夹路径
    
    返回:
        image_files: List[Path],按文件名排序的图像文件列表
    """
    image_files = []
    
    # 递归扫描所有 .png 文件
    for img in folder_path.rglob('*.png'):
        # 过滤条件:排除非原始图像
        exclude_keywords = ['ground_truth', 'temp_cropped', 'visualization', 
                          'result', 'debug', 'output']
        if all(keyword not in img.name.lower() for keyword in exclude_keywords):
            image_files.append(img)
    
    # 按文件名排序(保证处理顺序一致)
    return sorted(image_files)


def process_single_image(image_path, recognizer, processors, config):
    """
    处理单张图像的完整OCR流程
    
    参数:
        image_path: Path对象,图像路径
        recognizer: OCR识别器(共享)
        processors: dict,包含 decoder, deduplicator, conf_filter, grid_enhancer
        config: dict,配置参数
    
    返回:
        (success: bool, message: str, stats: dict)
    """
    try:
        # 1. 图像预处理与裁剪
        filename = image_path.name
        match = re.search(r'_0(\d)\.png$', filename)
        
        if match:
            page_num = int(match.group(1))
            if page_num in [1, 3]:
                crop_region = CROP_REGIONS['page1']
            elif page_num in [2, 4]:
                crop_region = CROP_REGIONS['page2']
            else:
                crop_region = None
        else:
            crop_region = None
        
        # 执行裁剪或使用原图
        if crop_region:
            cropped_img = crop_image(image_path, crop_region)
        else:
            cropped_img = Image.open(image_path)
        
        # 转换为OpenCV格式
        img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        
        # 2. 格子检测与分割
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = detect_grid_lines(image=gray, min_length=config['grid_min_length'])
        horizontal_lines = lines['horizontal']
        vertical_lines = lines['vertical']
        
        cells = generate_grid_cells(horizontal_lines, vertical_lines, img.shape[:2])
        
        # 过滤非空格子
        cell_images = []
        non_empty_cells = []
        
        for cell in cells:
            x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
            cell_img = img[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                h, w = gray_cell.shape
                crop = int(min(h, w) * config['center_crop_ratio'])
                
                if h > 2*crop and w > 2*crop:
                    center_region = gray_cell[crop:h-crop, crop:w-crop]
                else:
                    center_region = gray_cell
                
                non_white_ratio = (center_region < 240).sum() / center_region.size
                if non_white_ratio > config['non_white_threshold']:
                    cell_images.append(cell_img)
                    non_empty_cells.append(cell)
        
        if len(cell_images) == 0:
            return (False, "未检测到非空格子", {})
        
        # 3. OCR识别
        batch_raw_outputs = recognizer.batch_predict_with_raw_output(cell_images)
        
        results_list = []
        for raw_output in batch_raw_outputs:
            decoded = processors['decoder'](raw_output)
            deduped = processors['deduplicator'](decoded)
            filtered = processors['conf_filter'](deduped)
            results_list.append(filtered)
        
        prediction_data_full = restore_empty_cells(results_list, cells, non_empty_cells)
        
        # 4. BERT增强
        enhanced_results_list = processors['grid_enhancer'].enhance_grids(
            grid_results=results_list,
            grid_indices=None
        )
        
        enhanced_prediction_data_full = restore_empty_cells(
            enhanced_results_list, cells, non_empty_cells
        )
        
        # 5. 保存结果
        original_path = save_prediction_text(prediction_data_full, image_path, mode='original')
        enhanced_path = save_prediction_text(enhanced_prediction_data_full, image_path, mode='enhanced')
        
        # 统计信息
        changed_count = sum(1 for o, e in zip(results_list, enhanced_results_list) 
                          if o.get('text', '') != e.get('text', ''))
        
        stats = {
            'total_cells': len(cells),
            'non_empty_cells': len(cell_images),
            'enhanced_changed': changed_count,
            'horizontal_lines': len(horizontal_lines),
            'vertical_lines': len(vertical_lines)
        }
        
        return (True, f"成功处理", stats)
        
    except Exception as e:
        return (False, f"处理失败: {str(e)}", {})


def main(input_folder=None):
    """
    批量处理整个文件夹
    
    参数:
        input_folder: str | Path,输入文件夹路径
                      如果为None,使用默认路径 data/samples
    """
    print("="*80)
    print("端到端OCR批量处理流程")
    print("="*80)
    
    # 1. 确定输入文件夹
    if input_folder is None:
        input_folder = PROJECT_ROOT / 'data' / 'samples'
    else:
        input_folder = Path(input_folder)
    
    if not input_folder.exists():
        print(f"✗ 错误: 输入文件夹不存在: {input_folder}")
        return
    
    # 2. 初始化全局处理器(只初始化一次!)
    print("\n[1/3] 初始化OCR处理器...")
    print("-"*80)
    
    recognizer = CustomTextRecognizer(
        model_name=OCR_MODEL,
        device=DEFAULT_DEVICE
    )
    print(f"✓ OCR模型: {OCR_MODEL}")
    
    processors = {
        'decoder': TopKDecoder(k=TOP_K),
        'deduplicator': CTCDeduplicator(),
        'conf_filter': ConfidenceFilter(threshold=CONFIDENCE_THRESHOLD),
        'grid_enhancer': GridContextEnhancer(
            model_name=BERT_MODEL,
            device=BERT_DEVICE,
            context_window=BERT_CONTEXT_WINDOW,
            confidence_threshold=BERT_CONFIDENCE_THRESHOLD
        )
    }
    print(f"✓ BERT增强器: {BERT_MODEL}")
    
    config = {
        'grid_min_length': GRID_MIN_LENGTH,
        'non_white_threshold': NON_WHITE_THRESHOLD,
        'center_crop_ratio': CENTER_CROP_RATIO
    }
    
    # 3. 扫描文件
    print(f"\n[2/3] 扫描输入文件夹...")
    print("-"*80)
    print(f"输入路径: {input_folder}")
    
    image_files = scan_input_folder(input_folder)
    
    if len(image_files) == 0:
        print("✗ 未找到图像文件")
        return
    
    print(f"✓ 找到 {len(image_files)} 张图像待处理")
    
    # 4. 批量处理
    print(f"\n[3/3] 批量处理中...")
    print("="*80)
    
    success_count = 0
    failed_files = []
    all_stats = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        print("-"*60)
        
        success, message, stats = process_single_image(
            img_path, recognizer, processors, config
        )
        
        if success:
            success_count += 1
            all_stats.append(stats)
            print(f"✓ {message}")
            print(f"  格子: {stats['total_cells']} 总计, {stats['non_empty_cells']} 非空")
            print(f"  BERT改变: {stats['enhanced_changed']} 个")
        else:
            failed_files.append((img_path.name, message))
            print(f"✗ {message}")
    
    # 5. 总结报告
    print(f"\n{'='*80}")
    print("批量处理完成")
    print(f"{'='*80}")
    print(f"总计: {len(image_files)} 张")
    print(f"成功: {success_count} 张")
    print(f"失败: {len(failed_files)} 张")
    
    if all_stats:
        avg_cells = sum(s['total_cells'] for s in all_stats) / len(all_stats)
        avg_non_empty = sum(s['non_empty_cells'] for s in all_stats) / len(all_stats)
        avg_changed = sum(s['enhanced_changed'] for s in all_stats) / len(all_stats)
        
        print(f"\n平均统计:")
        print(f"  格子数: {avg_cells:.1f}")
        print(f"  非空格子: {avg_non_empty:.1f}")
        print(f"  BERT改变: {avg_changed:.1f}")
    
    if failed_files:
        print(f"\n失败文件列表:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print(f"\n输出目录: {PROJECT_ROOT / 'output' / 'predictions'}")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量OCR处理流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline_v3.py
  python pipeline_v3.py --input "data/samples/2022 第2題 (冬奧) (8份)_Original"
  python pipeline_v3.py -i "data/samples"
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='输入文件夹路径 (默认: data/samples)'
    )
    
    args = parser.parse_args()
    
    main(input_folder=args.input)