"""
BERT 参数网格搜索实验
===================

实验目标：
    寻找最佳的 BERT 权重和置信度阈值组合，以最大化 OCR 准确率

实验参数：
    - confidence_threshold: 0.3 ~ 1.0 (8个值)
    - bert_weight: 0.3 ~ 1.0 (8个值)
    - 总计: 8 × 8 = 64 次实验

输出：
    - CSV 报告: grid_search_results_YYYYMMDD_HHMMSS.csv
    - 可视化图表: grid_search_plots_YYYYMMDD_HHMMSS.png
    - 实验总结: grid_search_summary_YYYYMMDD_HHMMSS.txt

Author: OCR Pipeline Team
Date: 2025-11-28
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

# 实验参数网格
CONFIDENCE_THRESHOLDS = np.round(np.linspace(0.55, 1.0, 10), 2)  # [0.55, 0.60, 0.65, ..., 1.00]
BERT_WEIGHTS = np.round(np.linspace(0.40, 0.85, 10), 2)          # [0.40, 0.45, 0.50, ..., 0.85]

# 基线对照实验
BASELINE_EXPERIMENTS = [
    {'name': '原始OCR（无BERT）', 'conf_thresh': 2.0, 'bert_weight': 0.0},  # 双重保险：超大阈值跳过+权重0
    {'name': '纯BERT（无OCR融合）', 'conf_thresh': 0.0, 'bert_weight': 1.0},
    {'name': '当前默认配置', 'conf_thresh': 0.8, 'bert_weight': 0.4},
]

# 数据路径
DATA_CONFIG = {
    'image_path': project_root / 'output' / 'temp_cropped.png',
    'gt_file': project_root / 'data' / 'samples' / '2022 第2題 (冬奧) (8份)_Original' / 'sample_01_01_ground_truth.txt',
}

# 模型配置
MODEL_CONFIG = {
    'ocr_model': 'PP-OCRv5_server_rec',
    'bert_model': 'bert-base-chinese',
    'ocr_device': 'gpu:0',     # PaddleOCR 使用 PaddlePaddle 格式
    'bert_device': 'cuda:0',   # PyTorch BERT 使用 PyTorch 格式
    'context_window': 20,
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


def analyze_bert_effects(enhanced_cells, pred_cells, gt_cells):
    """分析 BERT 的改变效果"""
    total_changed = 0
    improved = 0
    degraded = 0
    
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
    
    return {
        'grids_changed': total_changed,
        'improved': improved,
        'degraded': degraded,
        'net_improvement': improved - degraded,
    }


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
                'row': row_idx, 'col': col_idx,
                'text': pred_text, 'confidence': conf,
                'is_empty': (pred_text == '')
            })
            gt_cells.append({
                'row': row_idx, 'col': col_idx,
                'text': gt_text
            })
    
    return pred_cells, gt_cells


# ==================== 主实验流程 ====================

def run_single_experiment(conf_thresh, bert_weight, enhancer_shared, 
                         results_list, cells, non_empty_cells, 
                         calculator, ground_truth, original_metrics):
    """运行单次实验"""
    start_time = time.time()
    
    # 创建新的增强器（不能共享，因为参数不同）
    enhancer = GridContextEnhancer(
        model_name=MODEL_CONFIG['bert_model'],
        device=MODEL_CONFIG['bert_device'],
        context_window=MODEL_CONFIG['context_window'],
        fusion_weight=1 - bert_weight,  # 注意：转换为 fusion_weight
        confidence_threshold=conf_thresh,
        verbose=False
    )
    
    # 如果阈值大于 1.0，跳过 BERT 处理（所有格子都不会被处理）
    if conf_thresh > 1.0:
        enhanced_results_list = results_list
        grids_processed = 0
    else:
        # BERT 增强
        enhanced_results_list = enhancer.enhance_grids(
            grid_results=results_list,
            grid_indices=None
        )
        
        # 统计处理的格子数
        grids_processed = sum(
            1 for r in results_list 
            if r.get('confidence', 1.0) < conf_thresh
        )
    
    # 恢复空格子
    enhanced_full = restore_empty_cells(enhanced_results_list, cells, non_empty_cells)
    
    # 计算准确率
    enhanced_metrics = calculator.calculate(
        predicted_results=enhanced_full,
        ground_truth=ground_truth,
        align_by_row=True
    )
    
    # 重构格子数据（用于分析 BERT 效果）
    enhanced_cells, _ = reconstruct_cell_data(enhanced_metrics)
    pred_cells, gt_cells = reconstruct_cell_data(original_metrics)
    
    # 分析 BERT 效果
    bert_effects = analyze_bert_effects(enhanced_cells, pred_cells, gt_cells)
    
    # 计算各项指标
    accuracy = enhanced_metrics['overall']['accuracy']
    cell_matches = sum(r['cell_matches'] for r in enhanced_metrics['by_row'])
    cell_total = sum(r['total_cells'] for r in enhanced_metrics['by_row'])
    cell_accuracy = (cell_matches / cell_total * 100) if cell_total > 0 else 0
    
    original_accuracy = original_metrics['overall']['accuracy']
    accuracy_gain = accuracy - original_accuracy
    relative_gain = (accuracy / original_accuracy - 1) * 100 if original_accuracy > 0 else 0
    
    processing_time = time.time() - start_time
    
    # 计算改进率
    improvement_rate = (bert_effects['improved'] / bert_effects['grids_changed'] * 100 
                       if bert_effects['grids_changed'] > 0 else 0)
    
    # 返回结果
    return {
        'conf_thresh': conf_thresh,
        'bert_weight': bert_weight,
        'accuracy': accuracy,
        'cell_accuracy': cell_accuracy,
        'grids_processed': grids_processed,
        'grids_changed': bert_effects['grids_changed'],
        'improved': bert_effects['improved'],
        'degraded': bert_effects['degraded'],
        'net_improvement': bert_effects['net_improvement'],
        'improvement_rate': improvement_rate,
        'accuracy_gain': accuracy_gain,
        'relative_gain': relative_gain,
        'processing_time': processing_time,
    }


def visualize_results(results_df, original_accuracy):
    """生成可视化图表"""
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建画布
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 准确率热力图
    ax1 = plt.subplot(2, 3, 1)
    accuracy_matrix = results_df.pivot(
        index='conf_thresh', 
        columns='bert_weight', 
        values='accuracy'
    )
    sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': '准确率 (%)'}, ax=ax1)
    ax1.set_xlabel('BERT 权重')
    ax1.set_ylabel('置信度阈值')
    ax1.set_title('字符准确率热力图')
    ax1.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax1.get_xticklabels()])
    ax1.set_yticklabels([f'{float(y.get_text()):.2f}' for y in ax1.get_yticklabels()])
    
    # 2. 净改进热力图
    ax2 = plt.subplot(2, 3, 2)
    improvement_matrix = results_df.pivot(
        index='conf_thresh', 
        columns='bert_weight', 
        values='net_improvement'
    )
    sns.heatmap(improvement_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                cbar_kws={'label': '净改进格子数'}, ax=ax2)
    ax2.set_xlabel('BERT 权重')
    ax2.set_ylabel('置信度阈值')
    ax2.set_title('净改进格子数热力图')
    ax2.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax2.get_xticklabels()])
    ax2.set_yticklabels([f'{float(y.get_text()):.2f}' for y in ax2.get_yticklabels()])
    
    # 3. 准确率提升散点图
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(
        results_df['accuracy_gain'],
        results_df['processing_time'],
        c=results_df['net_improvement'],
        s=100,
        cmap='RdYlGn',
        alpha=0.6
    )
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax3.set_xlabel('准确率提升 (%)')
    ax3.set_ylabel('处理时间 (秒)')
    ax3.set_title('成本效益分析（颜色=净改进）')
    plt.colorbar(scatter, ax=ax3, label='净改进格子数')
    
    # 标注最佳点
    best_idx = results_df['accuracy'].idxmax()
    best_row = results_df.iloc[best_idx]
    ax3.scatter(best_row['accuracy_gain'], best_row['processing_time'],
               c='red', s=300, marker='*', edgecolors='black', linewidths=2,
               label=f'最佳: conf={best_row["conf_thresh"]:.1f}, bert={best_row["bert_weight"]:.1f}')
    ax3.legend()
    
    # 4. 固定置信度阈值，看 BERT 权重的影响
    ax4 = plt.subplot(2, 3, 4)
    for conf_thresh in CONFIDENCE_THRESHOLDS[::2]:  # 每隔一个显示
        subset = results_df[results_df['conf_thresh'] == conf_thresh]
        ax4.plot(subset['bert_weight'], subset['accuracy'], 
                marker='o', label=f'conf={round(conf_thresh, 2):.2f}')
    ax4.axhline(y=original_accuracy, color='red', linestyle='--', 
               label='原始OCR基线', linewidth=2)
    ax4.set_xlabel('BERT 权重')
    ax4.set_ylabel('准确率 (%)')
    ax4.set_title('固定置信度阈值的准确率曲线')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 固定 BERT 权重，看置信度阈值的影响
    ax5 = plt.subplot(2, 3, 5)
    for bert_weight in BERT_WEIGHTS[::2]:  # 每隔一个显示
        subset = results_df[results_df['bert_weight'] == bert_weight]
        ax5.plot(subset['conf_thresh'], subset['accuracy'], 
                marker='o', label=f'bert={round(bert_weight, 2):.2f}')
    ax5.axhline(y=original_accuracy, color='red', linestyle='--', 
               label='原始OCR基线', linewidth=2)
    ax5.set_xlabel('置信度阈值')
    ax5.set_ylabel('准确率 (%)')
    ax5.set_title('固定BERT权重的准确率曲线')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 改进率统计
    ax6 = plt.subplot(2, 3, 6)
    improvement_rate_matrix = results_df.pivot(
        index='conf_thresh', 
        columns='bert_weight', 
        values='improvement_rate'
    )
    sns.heatmap(improvement_rate_matrix, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': '改进成功率 (%)'}, ax=ax6)
    ax6.set_xlabel('BERT 权重')
    ax6.set_ylabel('置信度阈值')
    ax6.set_title('BERT 改进成功率热力图')
    ax6.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax6.get_xticklabels()])
    ax6.set_yticklabels([f'{float(y.get_text()):.2f}' for y in ax6.get_yticklabels()])
    
    plt.tight_layout()
    
    # 保存图表
    output_path = OUTPUT_DIR / f'grid_search_plots_{TIMESTAMP}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可视化图表已保存: {output_path}")
    
    return output_path


def generate_summary_report(results_df, baseline_results, original_accuracy):
    """生成文本摘要报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BERT 参数网格搜索实验报告")
    report_lines.append("=" * 80)
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"实验次数: {len(results_df)} 次")
    report_lines.append(f"总耗时: {results_df['processing_time'].sum():.1f} 秒 ({results_df['processing_time'].sum()/60:.1f} 分钟)")
    
    # 基线结果
    report_lines.append(f"\n{'='*80}")
    report_lines.append("【基线对照实验】")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"{'实验名称':<25} {'准确率':<12} {'提升':<12} {'净改进':<12}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'原始OCR基线':<25} {original_accuracy:<12.2f}% {0.0:<12.2f}% {0:<12}")
    
    for baseline in baseline_results:
        name = baseline['name']
        acc = baseline['accuracy']
        gain = baseline['accuracy_gain']
        net = baseline['net_improvement']
        report_lines.append(f"{name:<25} {acc:<12.2f}% {gain:<12.2f}% {net:<12}")
    
    # 最佳参数组合
    report_lines.append(f"\n{'='*80}")
    report_lines.append("【最佳参数组合】")
    report_lines.append(f"{'='*80}")
    
    # 按准确率排序
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    report_lines.append(f"\n1. 最高准确率:")
    report_lines.append(f"   置信度阈值: {best_accuracy['conf_thresh']:.2f}")
    report_lines.append(f"   BERT 权重: {best_accuracy['bert_weight']:.2f}")
    report_lines.append(f"   准确率: {best_accuracy['accuracy']:.2f}%")
    report_lines.append(f"   提升: {best_accuracy['accuracy_gain']:.2f}%")
    report_lines.append(f"   净改进: {best_accuracy['net_improvement']} 个格子")
    
    # 按净改进排序
    best_improvement = results_df.loc[results_df['net_improvement'].idxmax()]
    report_lines.append(f"\n2. 最大净改进:")
    report_lines.append(f"   置信度阈值: {best_improvement['conf_thresh']:.2f}")
    report_lines.append(f"   BERT 权重: {best_improvement['bert_weight']:.2f}")
    report_lines.append(f"   净改进: {best_improvement['net_improvement']} 个格子")
    report_lines.append(f"   准确率: {best_improvement['accuracy']:.2f}%")
    report_lines.append(f"   提升: {best_improvement['accuracy_gain']:.2f}%")
    
    # 按性价比（准确率提升/时间）排序
    results_df['efficiency'] = results_df['accuracy_gain'] / results_df['processing_time']
    best_efficiency = results_df.loc[results_df['efficiency'].idxmax()]
    report_lines.append(f"\n3. 最高性价比:")
    report_lines.append(f"   置信度阈值: {best_efficiency['conf_thresh']:.2f}")
    report_lines.append(f"   BERT 权重: {best_efficiency['bert_weight']:.2f}")
    report_lines.append(f"   准确率: {best_efficiency['accuracy']:.2f}%")
    report_lines.append(f"   处理时间: {best_efficiency['processing_time']:.1f} 秒")
    report_lines.append(f"   效率: {best_efficiency['efficiency']:.4f} %/秒")
    
    # 统计分析
    report_lines.append(f"\n{'='*80}")
    report_lines.append("【统计分析】")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"\n准确率统计:")
    report_lines.append(f"   最高: {results_df['accuracy'].max():.2f}%")
    report_lines.append(f"   最低: {results_df['accuracy'].min():.2f}%")
    report_lines.append(f"   平均: {results_df['accuracy'].mean():.2f}%")
    report_lines.append(f"   标准差: {results_df['accuracy'].std():.2f}%")
    
    report_lines.append(f"\n准确率提升统计:")
    report_lines.append(f"   最高: {results_df['accuracy_gain'].max():.2f}%")
    report_lines.append(f"   最低: {results_df['accuracy_gain'].min():.2f}%")
    report_lines.append(f"   平均: {results_df['accuracy_gain'].mean():.2f}%")
    report_lines.append(f"   正提升比例: {(results_df['accuracy_gain'] > 0).sum() / len(results_df) * 100:.1f}%")
    
    report_lines.append(f"\n净改进统计:")
    report_lines.append(f"   最高: {results_df['net_improvement'].max()}")
    report_lines.append(f"   最低: {results_df['net_improvement'].min()}")
    report_lines.append(f"   平均: {results_df['net_improvement'].mean():.1f}")
    report_lines.append(f"   正改进比例: {(results_df['net_improvement'] > 0).sum() / len(results_df) * 100:.1f}%")
    
    # 参数趋势分析
    report_lines.append(f"\n{'='*80}")
    report_lines.append("【参数趋势分析】")
    report_lines.append(f"{'='*80}")
    
    # 置信度阈值的影响
    conf_corr = results_df[['conf_thresh', 'accuracy']].corr().iloc[0, 1]
    report_lines.append(f"\n置信度阈值与准确率的相关性: {conf_corr:.3f}")
    
    # BERT 权重的影响
    bert_corr = results_df[['bert_weight', 'accuracy']].corr().iloc[0, 1]
    report_lines.append(f"BERT 权重与准确率的相关性: {bert_corr:.3f}")
    
    # Top 5 和 Bottom 5
    report_lines.append(f"\n{'='*80}")
    report_lines.append("【Top 5 参数组合】")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"{'排名':<6} {'置信度':<10} {'BERT权重':<10} {'准确率':<12} {'提升':<12}")
    report_lines.append("-" * 80)
    
    top5 = results_df.nlargest(5, 'accuracy')
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        report_lines.append(
            f"{i:<6} {row['conf_thresh']:<10.2f} {row['bert_weight']:<10.2f} "
            f"{row['accuracy']:<12.2f}% {row['accuracy_gain']:<12.2f}%"
        )
    
    report_lines.append(f"\n{'='*80}")
    report_lines.append("【Bottom 5 参数组合】")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"{'排名':<6} {'置信度':<10} {'BERT权重':<10} {'准确率':<12} {'提升':<12}")
    report_lines.append("-" * 80)
    
    bottom5 = results_df.nsmallest(5, 'accuracy')
    for i, (idx, row) in enumerate(bottom5.iterrows(), 1):
        report_lines.append(
            f"{i:<6} {row['conf_thresh']:<10.2f} {row['bert_weight']:<10.2f} "
            f"{row['accuracy']:<12.2f}% {row['accuracy_gain']:<12.2f}%"
        )
    
    report_lines.append(f"\n{'='*80}")
    report_lines.append("实验完成")
    report_lines.append(f"{'='*80}\n")
    
    # 保存报告
    report_text = "\n".join(report_lines)
    output_path = OUTPUT_DIR / f'grid_search_summary_{TIMESTAMP}.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ 实验摘要已保存: {output_path}")
    
    # 同时打印到控制台
    print(report_text)
    
    return output_path


def main():
    """主函数"""
    print("=" * 80)
    print("BERT 参数网格搜索实验")
    print("=" * 80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # ==================== 1. 加载数据 ====================
    print("\n" + "=" * 80)
    print("步骤 1: 加载数据")
    print("=" * 80)
    
    # 加载图像
    img_path = DATA_CONFIG['image_path']
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {img_path}")
    print(f"✓ 加载图像: {img.shape}")
    
    # 加载 Ground Truth
    gt_file = DATA_CONFIG['gt_file']
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = f.read()
    
    # 清理 ground truth
    gt_clean = re.sub(r'<insert>|</insert>|<\?>', '', ground_truth)
    gt_clean = gt_clean.replace('\n', '').replace(' ', '')
    print(f"✓ 加载 Ground Truth: {len(gt_clean)} 字符")
    
    # ==================== 2. 格子检测 ====================
    print("\n" + "=" * 80)
    print("步骤 2: 格子检测与分割")
    print("=" * 80)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = detect_grid_lines(gray)
    cells = generate_grid_cells(lines['horizontal'], lines['vertical'], img.shape[:2])
    print(f"✓ 生成格子: {len(cells)} 个")
    
    # 过滤非空格子
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
    
    # ==================== 3. 原始 OCR ====================
    print("\n" + "=" * 80)
    print("步骤 3: 原始 OCR 识别")
    print("="  * 80)
    
    recognizer = CustomTextRecognizer(
        model_name=MODEL_CONFIG['ocr_model'],
        device=MODEL_CONFIG['ocr_device']
    )
    
    decoder = TopKDecoder(k=5)
    deduplicator = CTCDeduplicator()
    conf_filter = ConfidenceFilter(threshold=0.3)
    
    batch_raw_outputs = recognizer.batch_predict_with_raw_output(cell_images)
    print(f"✓ 获取 {len(batch_raw_outputs)} 个格子的概率矩阵")
    
    results_list = []
    for raw_output in batch_raw_outputs:
        decoded = decoder(raw_output)
        deduped = deduplicator(decoded)
        filtered = conf_filter(deduped)
        results_list.append(filtered)
    
    prediction_data_full = restore_empty_cells(results_list, cells, non_empty_cells)
    print(f"✓ 完整预测结果: {len(prediction_data_full)} 个格子")
    
    # 计算原始准确率
    calculator = GridAccuracyCalculator(empty_char='')
    original_metrics = calculator.calculate(
        predicted_results=prediction_data_full,
        ground_truth=ground_truth,
        align_by_row=True
    )
    
    original_accuracy = original_metrics['overall']['accuracy']
    print(f"✓ 原始 OCR 准确率: {original_accuracy:.2f}%")
    
    # ==================== 4. 基线对照实验 ====================
    print("\n" + "=" * 80)
    print("步骤 4: 基线对照实验")
    print("=" * 80)
    
    baseline_results = []
    for baseline in BASELINE_EXPERIMENTS:
        print(f"\n运行基线实验: {baseline['name']}")
        result = run_single_experiment(
            conf_thresh=baseline['conf_thresh'],
            bert_weight=baseline['bert_weight'],
            enhancer_shared=None,
            results_list=results_list,
            cells=cells,
            non_empty_cells=non_empty_cells,
            calculator=calculator,
            ground_truth=ground_truth,
            original_metrics=original_metrics
        )
        result['name'] = baseline['name']
        baseline_results.append(result)
        print(f"  准确率: {result['accuracy']:.2f}% (提升: {result['accuracy_gain']:.2f}%)")
    
    # ==================== 5. 网格搜索实验 ====================
    print("\n" + "=" * 80)
    print("步骤 5: 网格搜索实验 (8 × 8 = 64 次)")
    print("=" * 80)
    
    results_list_exp = []
    total_experiments = len(CONFIDENCE_THRESHOLDS) * len(BERT_WEIGHTS)
    
    with tqdm(total=total_experiments, desc="实验进度") as pbar:
        for conf_thresh in CONFIDENCE_THRESHOLDS:
            for bert_weight in BERT_WEIGHTS:
                result = run_single_experiment(
                    conf_thresh=conf_thresh,
                    bert_weight=bert_weight,
                    enhancer_shared=None,
                    results_list=results_list,
                    cells=cells,
                    non_empty_cells=non_empty_cells,
                    calculator=calculator,
                    ground_truth=ground_truth,
                    original_metrics=original_metrics
                )
                results_list_exp.append(result)
                pbar.update(1)
                
                # 实时保存（避免中断丢失数据）
                results_df = pd.DataFrame(results_list_exp)
                csv_path = OUTPUT_DIR / f'grid_search_results_{TIMESTAMP}.csv'
                results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ 所有实验完成!")
    
    # ==================== 6. 保存结果 ====================
    print("\n" + "=" * 80)
    print("步骤 6: 保存结果")
    print("=" * 80)
    
    results_df = pd.DataFrame(results_list_exp)
    
    # 保存 CSV
    csv_path = OUTPUT_DIR / f'grid_search_results_{TIMESTAMP}.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV 结果已保存: {csv_path}")
    
    # 生成可视化
    plot_path = visualize_results(results_df, original_accuracy)
    
    # 生成摘要报告
    summary_path = generate_summary_report(results_df, baseline_results, original_accuracy)
    
    # ==================== 7. 完成 ====================
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {results_df['processing_time'].sum():.1f} 秒 ({results_df['processing_time'].sum()/60:.1f} 分钟)")
    print(f"\n输出文件:")
    print(f"  - CSV: {csv_path}")
    print(f"  - 图表: {plot_path}")
    print(f"  - 摘要: {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
