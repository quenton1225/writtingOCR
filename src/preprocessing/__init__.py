"""
Preprocessing utilities for essay OCR

包含竖线检测、格子检测等预处理模块
"""

from .line_detection import (
    detect_vertical_lines,
    visualize_vertical_lines,
    get_vertical_lines_statistics
)

from .line_merging import (
    merge_vertical_lines,
    get_merge_statistics,
    visualize_merge_comparison
)

from .line_removal import (
    remove_vertical_lines,
    remove_vertical_lines_iterative,
    visualize_removal_comparison
)

from .grid_detection import (
    detect_grid_lines,
    cluster_lines_by_position,
    merge_line_cluster,
    generate_grid_cells,
    is_cell_empty,
    visualize_grid,
    get_grid_statistics
)

__all__ = [
    # 竖线检测
    'detect_vertical_lines',
    'visualize_vertical_lines',
    'get_vertical_lines_statistics',
    
    # 竖线合并
    'merge_vertical_lines',
    'get_merge_statistics',
    'visualize_merge_comparison',
    
    # 竖线删除
    'remove_vertical_lines',
    'remove_vertical_lines_iterative',
    'visualize_removal_comparison',
    
    # 格子检测
    'detect_grid_lines',
    'cluster_lines_by_position',
    'merge_line_cluster',
    'generate_grid_cells',
    'is_cell_empty',
    'visualize_grid',
    'get_grid_statistics',
]
