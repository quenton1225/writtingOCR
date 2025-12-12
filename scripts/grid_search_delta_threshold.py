"""
Delta Threshold 网格搜索实验
===========================

实验目标：
    在最优参数基础上(conf_thresh=0.9, fusion_weight=0.45)，
    搜索最佳的 delta_threshold 值

实验参数：
    - confidence_threshold: 0.9 (固定)
    - fusion_weight: 0.45 (固定，对应 bert_weight=0.55)
    - delta_threshold: [-0.5, -0.35, -0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.35, 0.5] (11个值)

输出：
    - CSV 报告: delta_search_results_YYYYMMDD_HHMMSS.csv
    - 可视化图表: delta_search_plots_YYYYMMDD_HHMMSS.png
    - 实验总结: delta_search_summary_YYYYMMDD_HHMMSS.txt

Author: OCR Pipeline Team
Date: 2025-11-29
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import pandas as pd
import cv2
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入自定义模块
from src.custom_ocr import CustomTextRecognizer
from src.custom_ocr.processors import TopKDecoder, CTCDeduplicator, ConfidenceFilter
from src.custom_ocr.processors.grid_context_enhancer import GridContextEnhancer
from src.preprocessing.grid_detection import detect_grid_lines, generate_grid_cells
from src.evaluation import GridAccuracyCalculator


# ==================== 配置区 ====================

# 固定的最优参数
FIXED_CONFIDENCE_THRESHOLD = 0.9
FIXED_FUSION_WEIGHT = 0.45  # 对应 bert_weight = 0.55

# 实验参数: delta_threshold
DELTA_THRESHOLDS = [-0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]

# 数据路径
DATA_CONFIG = {
    'image_path': project_root / 'output' / 'temp_cropped.png',
    'gt_file': project_root / 'data' / 'samples' / '2022 第2題 (冬奧) (8份)_Original' / 'sample_01_01_ground_truth.txt',
}

# 模型配置
MODEL_CONFIG = {
    'ocr_model': 'PP-OCRv5_server_rec',
    'bert_model': 'bert-base-chinese',
    'ocr_device': 'gpu:0',
    'bert_device': 'cuda:0',
    'context_window': 10,
}

# 输出配置
OUTPUT_DIR = project_root / 'output' / 'grid_search'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 时间戳
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# ==================== 辅助函数 ====================

def restore_empty_cells(results_list, cells, non_empty_cells):
    """恢复空格子，构建完整的预测结果"""
    all_cells_dict = {}
    for cell in cells:
        row, col = cell['row'], cell['col']
        all_cells_dict[(row, col)] = {
            'row': row, 'col': col, 'text': '',
            'confidence': 1.0, 'is_empty': True
        }
    
    for i, (result, cell) in enumerate(zip(results_list, non_empty_cells)):
        row, col = cell['row'], cell['col']
        all_cells_dict[(row, col)] = {
            'row': row, 'col': col,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'is_empty': False,
            'ocr_result': result
        }
    
    prediction_data_full = sorted(all_cells_dict.values(), 
                                  key=lambda x: (x['row'], x['col']))
    return prediction_data_full


def reconstruct_cell_data(metrics_dict):
    """从 metrics 重构格子数据"""
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


def analyze_bert_effects(enhanced_cells, pred_cells, gt_cells):
    """分析 BERT 的改变效果"""
    total_changed = 0
    improved = 0
    degraded = 0
    wrong_to_wrong = 0
    
    for pred, enhanced, gt in zip(pred_cells, enhanced_cells, gt_cells):
        if pred.get('is_empty', False):
            continue
        
        pred_text = pred['text']
        enhanced_text = enhanced['text']
        gt_text = gt['text']
        
        if pred_text != enhanced_text:
            total_changed += 1
            pred_correct = (pred_text == gt_text)
            enhanced_correct = (enhanced_text == gt_text)
            
            if not pred_correct and enhanced_correct:
                improved += 1
            elif pred_correct and not enhanced_correct:
                degraded += 1
            else:
                wrong_to_wrong += 1
    
    return {
        'triggered': total_changed,
        'improved': improved,
        'degraded': degraded,
        'wrong_to_wrong': wrong_to_wrong,
        'net_improvement': improved - degraded,
    }


def run_single_experiment(delta_threshold, baseline_data):
    """运行单次实验"""
    print(f"\n{'='*80}")
    print(f"实验: delta_threshold={delta_threshold}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 解包基线数据
    recognizer = baseline_data['recognizer']
    cells = baseline_data['cells']
    non_empty_cells = baseline_data['non_empty_cells']
    cell_images = baseline_data['cell_images']
    results_list = baseline_data['results_list']
    ground_truth = baseline_data['ground_truth']
    calculator = baseline_data['calculator']
    original_metrics = baseline_data['original_metrics']
    
    # 创建 BERT 增强器 (使用新的 delta_threshold 参数)
    enhancer = GridContextEnhancer(
        model_name=MODEL_CONFIG['bert_model'],
        device=MODEL_CONFIG['bert_device'],
        context_window=MODEL_CONFIG['context_window'],
        confidence_threshold=FIXED_CONFIDENCE_THRESHOLD,
        fusion_weight=FIXED_FUSION_WEIGHT,
        delta_threshold=delta_threshold,  # 🔥 关键参数
        verbose=False
    )
    
    # 批量增强
    enhanced_results_list = enhancer.enhance_grids(
        grid_results=results_list,
        grid_indices=None  # 自动识别低置信度格子
    )
    
    # 统计触发数
    triggered_count = sum(
        1 for r in enhanced_results_list
        if r.get('grid_bert_correction', {}).get('corrected', False)
    )
    
    # 恢复空格子
    enhanced_full = restore_empty_cells(enhanced_results_list, cells, non_empty_cells)
    
    # 计算准确率
    enhanced_metrics = calculator.calculate(
        predicted_results=enhanced_full,
        ground_truth=ground_truth,
        align_by_row=True
    )
    
    # 重构格子数据
    enhanced_cells, _ = reconstruct_cell_data(enhanced_metrics)
    pred_cells, gt_cells = reconstruct_cell_data(original_metrics)
    
    # 分析 BERT 效果
    bert_effects = analyze_bert_effects(enhanced_cells, pred_cells, gt_cells)
    
    # 计算指标
    accuracy = enhanced_metrics['overall']['accuracy']
    original_accuracy = original_metrics['overall']['accuracy']
    accuracy_gain = accuracy - original_accuracy
    
    non_empty_count = len([c for c in pred_cells if not c.get('is_empty', False)])
    trigger_rate = (bert_effects['triggered'] / non_empty_count * 100) if non_empty_count > 0 else 0
    correction_rate = (bert_effects['improved'] / bert_effects['triggered'] * 100) if bert_effects['triggered'] > 0 else 0
    error_rate = (bert_effects['degraded'] / bert_effects['triggered'] * 100) if bert_effects['triggered'] > 0 else 0
    
    processing_time = time.time() - start_time
    
    # 打印结果
    print(f"\n结果:")
    print(f"  准确率: {accuracy:.2f}% (提升: {accuracy_gain:+.2f}%)")
    print(f"  触发率: {trigger_rate:.2f}% ({bert_effects['triggered']}/{non_empty_count})")
    print(f"  纠正率: {correction_rate:.2f}% ({bert_effects['improved']}个)")
    print(f"  错误率: {error_rate:.2f}% ({bert_effects['degraded']}个)")
    print(f"  净改进: {bert_effects['net_improvement']} 个格子")
    print(f"  处理时间: {processing_time:.1f}秒")
    
    # 返回结果
    return {
        'delta_threshold': delta_threshold,
        'accuracy': accuracy,
        'accuracy_gain': accuracy_gain,
        'triggered': bert_effects['triggered'],
        'trigger_rate': trigger_rate,
        'improved': bert_effects['improved'],
        'degraded': bert_effects['degraded'],
        'wrong_to_wrong': bert_effects['wrong_to_wrong'],
        'net_improvement': bert_effects['net_improvement'],
        'correction_rate': correction_rate,
        'error_rate': error_rate,
        'processing_time': processing_time,
    }


def prepare_baseline_data():
    """准备基线数据 (OCR 和格子检测)"""
    print("="*80)
    print("准备基线数据...")
    print("="*80)
    
    # 1. 初始化 OCR 识别器
    print("\n初始化 OCR 识别器...")
    recognizer = CustomTextRecognizer(
        model_name=MODEL_CONFIG['ocr_model'],
        device=MODEL_CONFIG['ocr_device']
    )
    
    # 2. 加载图像和 ground truth
    print("加载测试数据...")
    img = cv2.imread(str(DATA_CONFIG['image_path']))
    with open(DATA_CONFIG['gt_file'], 'r', encoding='utf-8') as f:
        ground_truth = f.read()
    
    # 3. 格子检测
    print("检测格子...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = detect_grid_lines(gray)
    cells = generate_grid_cells(lines['horizontal'], lines['vertical'], img.shape[:2])
    
    # 4. 过滤非空格子
    print("过滤非空格子...")
    cell_images = []
    non_empty_cells = []
    for cell in cells:
        x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
        cell_img = img[y1:y2, x1:x2]
        if cell_img.size > 0:
            gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            h, w = gray_cell.shape
            crop = int(min(h, w) * 0.08)
            if h > 2*crop and w > 2*crop:
                center_region = gray_cell[crop:h-crop, crop:w-crop]
            else:
                center_region = gray_cell
            non_white_ratio = (center_region < 240).sum() / center_region.size
            if non_white_ratio > 0.005:
                cell_images.append(cell_img)
                non_empty_cells.append(cell)
    
    print(f"✓ 非空格子数: {len(cell_images)}")
    
    # 5. OCR 识别
    print("\nOCR 识别所有格子...")
    decoder = TopKDecoder(k=5)
    deduplicator = CTCDeduplicator()
    conf_filter = ConfidenceFilter(threshold=0.3)
    
    batch_raw_outputs = recognizer.batch_predict_with_raw_output(cell_images)
    results_list = []
    for raw_output in batch_raw_outputs:
        decoded = decoder(raw_output)
        deduped = deduplicator(decoded)
        filtered = conf_filter(deduped)
        results_list.append(filtered)
    
    prediction_data_full = restore_empty_cells(results_list, cells, non_empty_cells)
    
    # 6. 计算原始准确率
    print("\n计算原始 OCR 准确率...")
    calculator = GridAccuracyCalculator(empty_char='')
    original_metrics = calculator.calculate(
        predicted_results=prediction_data_full,
        ground_truth=ground_truth,
        align_by_row=True
    )
    
    original_accuracy = original_metrics['overall']['accuracy']
    print(f"✓ 原始 OCR 准确率: {original_accuracy:.2f}%")
    
    # 返回所有基线数据
    return {
        'recognizer': recognizer,
        'cells': cells,
        'non_empty_cells': non_empty_cells,
        'cell_images': cell_images,
        'results_list': results_list,
        'ground_truth': ground_truth,
        'calculator': calculator,
        'original_metrics': original_metrics,
        'original_accuracy': original_accuracy,
    }


def visualize_results(results_df, original_accuracy):
    """生成可视化图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 准确率 vs delta
    ax = axes[0, 0]
    ax.plot(results_df['delta_threshold'], results_df['accuracy'], 
            marker='o', linewidth=2, markersize=10, color='blue')
    ax.axhline(y=original_accuracy, color='red', linestyle='--', 
              label=f'原始OCR基线 ({original_accuracy:.2f}%)', linewidth=2)
    best_idx = results_df['accuracy'].idxmax()
    best_row = results_df.iloc[best_idx]
    ax.scatter(best_row['delta_threshold'], best_row['accuracy'],
              c='red', s=300, marker='*', edgecolors='black', linewidths=2,
              label=f'最佳: δ={best_row["delta_threshold"]:.2f}')
    ax.set_xlabel('Delta Threshold')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('准确率 vs Delta Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 触发率 vs delta
    ax = axes[0, 1]
    ax.plot(results_df['delta_threshold'], results_df['trigger_rate'], 
            marker='s', linewidth=2, markersize=10, color='green')
    ax.set_xlabel('Delta Threshold')
    ax.set_ylabel('触发率 (%)')
    ax.set_title('BERT 触发率 vs Delta')
    ax.grid(True, alpha=0.3)
    
    # 3. 纠正率和错误率
    ax = axes[0, 2]
    ax.plot(results_df['delta_threshold'], results_df['correction_rate'], 
            marker='o', linewidth=2, label='纠正率', color='green')
    ax.plot(results_df['delta_threshold'], results_df['error_rate'], 
            marker='s', linewidth=2, label='错误引入率', color='red')
    ax.set_xlabel('Delta Threshold')
    ax.set_ylabel('比率 (%)')
    ax.set_title('纠正率 vs 错误引入率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 净改进 vs delta
    ax = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in results_df['net_improvement']]
    ax.bar(results_df['delta_threshold'].astype(str), results_df['net_improvement'], 
          color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Delta Threshold')
    ax.set_ylabel('净改进格子数')
    ax.set_title('净改进 vs Delta')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. 改对/改错对比
    ax = axes[1, 1]
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width/2, results_df['improved'], width, label='改对', color='green', alpha=0.7)
    ax.bar(x + width/2, results_df['degraded'], width, label='改错', color='red', alpha=0.7)
    ax.set_xlabel('Delta Threshold')
    ax.set_ylabel('格子数')
    ax.set_title('改对 vs 改错')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.2f}' for d in results_df['delta_threshold']])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. 综合效率指标
    ax = axes[1, 2]
    # 计算效率: 准确率提升 / 触发率 (每触发1%能带来多少准确率提升)
    efficiency = results_df['accuracy_gain'] / (results_df['trigger_rate'] + 0.01)  # 避免除0
    ax.plot(results_df['delta_threshold'], efficiency, 
            marker='D', linewidth=2, markersize=10, color='purple')
    ax.set_xlabel('Delta Threshold')
    ax.set_ylabel('效率 (准确率提升/触发率)')
    ax.set_title('BERT 使用效率')
    ax.grid(True, alpha=0.3)
    # 将纵坐标设置为对数刻度
    ax.set_yscale('log', base=10)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = OUTPUT_DIR / f'delta_search_plots_{TIMESTAMP}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {plot_file}")
    
    return fig


def generate_summary(results_df, original_accuracy, total_time):
    """生成实验总结报告"""
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("Delta Threshold 网格搜索实验报告")
    summary_lines.append("="*80)
    summary_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"实验次数: {len(results_df)} 次")
    summary_lines.append(f"总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("【固定参数】")
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"置信度阈值: {FIXED_CONFIDENCE_THRESHOLD}")
    summary_lines.append(f"融合权重 (fusion_weight): {FIXED_FUSION_WEIGHT} (对应 bert_weight={1-FIXED_FUSION_WEIGHT:.2f})")
    summary_lines.append(f"原始 OCR 准确率: {original_accuracy:.2f}%")
    
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("【最佳配置】")
    summary_lines.append(f"{'='*80}")
    
    # 最高准确率
    best_acc_idx = results_df['accuracy'].idxmax()
    best_acc = results_df.iloc[best_acc_idx]
    summary_lines.append(f"\n1. 最高准确率:")
    summary_lines.append(f"   Delta: {best_acc['delta_threshold']:.2f}")
    summary_lines.append(f"   准确率: {best_acc['accuracy']:.2f}%")
    summary_lines.append(f"   提升: {best_acc['accuracy_gain']:+.2f}%")
    summary_lines.append(f"   净改进: {best_acc['net_improvement']} 个格子")
    summary_lines.append(f"   触发率: {best_acc['trigger_rate']:.2f}%")
    summary_lines.append(f"   错误率: {best_acc['error_rate']:.2f}%")
    
    # 最大净改进
    best_net_idx = results_df['net_improvement'].idxmax()
    best_net = results_df.iloc[best_net_idx]
    summary_lines.append(f"\n2. 最大净改进:")
    summary_lines.append(f"   Delta: {best_net['delta_threshold']:.2f}")
    summary_lines.append(f"   净改进: {best_net['net_improvement']} 个格子")
    summary_lines.append(f"   准确率: {best_net['accuracy']:.2f}%")
    summary_lines.append(f"   提升: {best_net['accuracy_gain']:+.2f}%")
    
    # 最低错误率
    best_err_idx = results_df['error_rate'].idxmin()
    best_err = results_df.iloc[best_err_idx]
    summary_lines.append(f"\n3. 最低错误引入率:")
    summary_lines.append(f"   Delta: {best_err['delta_threshold']:.2f}")
    summary_lines.append(f"   错误率: {best_err['error_rate']:.2f}%")
    summary_lines.append(f"   准确率: {best_err['accuracy']:.2f}%")
    summary_lines.append(f"   净改进: {best_err['net_improvement']} 个格子")
    
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("【完整结果表】")
    summary_lines.append(f"{'='*80}")
    summary_lines.append(f"{'Delta':<10} {'准确率':<10} {'提升':<10} {'触发率':<10} {'纠正率':<10} {'错误率':<10} {'净改进':<10}")
    summary_lines.append("-"*80)
    for _, row in results_df.iterrows():
        summary_lines.append(
            f"{row['delta_threshold']:<10.2f} {row['accuracy']:<10.2f}% "
            f"{row['accuracy_gain']:<10.2f}% {row['trigger_rate']:<10.2f}% "
            f"{row['correction_rate']:<10.2f}% {row['error_rate']:<10.2f}% "
            f"{row['net_improvement']:<10.0f}"
        )
    
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("【关键发现】")
    summary_lines.append(f"{'='*80}")
    
    # 分析趋势
    if results_df['accuracy'].is_monotonic_increasing:
        summary_lines.append("• Delta 越大,准确率越高 (正相关)")
    elif results_df['accuracy'].is_monotonic_decreasing:
        summary_lines.append("• Delta 越大,准确率越低 (负相关)")
    else:
        summary_lines.append("• 准确率呈现非单调关系,存在最优点")
    
    # 触发率变化
    trigger_diff = results_df['trigger_rate'].max() - results_df['trigger_rate'].min()
    summary_lines.append(f"• 触发率变化范围: {results_df['trigger_rate'].min():.2f}% ~ {results_df['trigger_rate'].max():.2f}% (Δ={trigger_diff:.2f}%)")
    
    # 错误率趋势
    if results_df['error_rate'].corr(results_df['delta_threshold']) > 0.5:
        summary_lines.append("• Delta 越大,错误引入率越高")
    elif results_df['error_rate'].corr(results_df['delta_threshold']) < -0.5:
        summary_lines.append("• Delta 越大,错误引入率越低")
    
    summary_lines.append(f"\n{'='*80}")
    summary_lines.append("实验完成")
    summary_lines.append(f"{'='*80}")
    
    summary_text = '\n'.join(summary_lines)
    
    # 保存到文件
    summary_file = OUTPUT_DIR / f'delta_search_summary_{TIMESTAMP}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n✓ 总结报告已保存: {summary_file}")
    
    return summary_text


def main():
    """主函数"""
    print("="*80)
    print("Delta Threshold 网格搜索实验")
    print("="*80)
    print(f"\n固定参数:")
    print(f"  confidence_threshold: {FIXED_CONFIDENCE_THRESHOLD}")
    print(f"  fusion_weight: {FIXED_FUSION_WEIGHT} (bert_weight={1-FIXED_FUSION_WEIGHT:.2f})")
    print(f"\n实验参数:")
    print(f"  delta_threshold: {DELTA_THRESHOLDS}")
    print(f"  实验次数: {len(DELTA_THRESHOLDS)}")
    
    # 准备基线数据
    start_time = time.time()
    baseline_data = prepare_baseline_data()
    original_accuracy = baseline_data['original_accuracy']
    
    # 运行实验
    results = []
    for delta in tqdm(DELTA_THRESHOLDS, desc="运行实验"):
        result = run_single_experiment(delta, baseline_data)
        results.append(result)
    
    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存 CSV
    csv_file = OUTPUT_DIR / f'delta_search_results_{TIMESTAMP}.csv'
    results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV 结果已保存: {csv_file}")
    
    # 生成可视化
    visualize_results(results_df, original_accuracy)
    
    # 生成总结报告
    total_time = time.time() - start_time
    generate_summary(results_df, original_accuracy, total_time)
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)


if __name__ == '__main__':
    main()
