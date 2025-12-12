"""
提取测试样本
从Original文件夹中提取少量作文作为测试样本
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.pdf_to_images import PDFConverter

# 导入PyMuPDF
try:
    import fitz
except ImportError:
    import pymupdf as fitz


def extract_test_samples():
    """提取测试样本"""
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 输入输出路径
    original_dir = project_root / "data" / "raw" / "Original"
    samples_dir = project_root / "data" / "samples"
    
    # 选择要提取的PDF(小样本测试)
    test_pdfs = [
        "2022 第2題 (冬奧) (8份)_Original.pdf",  # 8份,样本少
        "2023 第2題 (藝術節) (13份)_Original.pdf",  # 13份
    ]
    
    print("=" * 60)
    print("提取测试样本")
    print("=" * 60)
    
    # 不合并页面，每页独立保存
    converter = PDFConverter(dpi=300, merge_pages=False)
    
    total_images = 0
    for pdf_name in test_pdfs:
        pdf_path = original_dir / pdf_name
        
        if not pdf_path.exists():
            print(f"⚠ 警告: {pdf_name} 不存在,跳过")
            continue
        
        # 提取图片（每页独立）
        output_dir = samples_dir / pdf_path.stem
        
        # 打开PDF获取总页数
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
        
        # 转换时使用自定义逻辑命名
        images = []
        doc = fitz.open(str(pdf_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for page_num in range(total_pages):
            # 计算是第几份作文的第几页
            essay_num = page_num // 2 + 1  # 每两页为一份作文
            page_in_essay = (page_num % 2) + 1  # 该作文的第几页(01或02)
            
            # 文件名格式: sample_01_01.png, sample_01_02.png, sample_02_01.png...
            filename = f"sample_{essay_num:02d}_{page_in_essay:02d}.png"
            
            # 保存页面
            page = doc.load_page(page_num)
            mat = fitz.Matrix(converter.zoom, converter.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            output_path = output_dir / filename
            pix.save(str(output_path))
            images.append(str(output_path))
        
        doc.close()
        total_images += len(images)
        print(f"✓ 提取 {len(images)} 页图片 (共 {len(images)//2} 份作文)")
    
    print("=" * 60)
    print(f"完成! 共提取 {total_images} 页图片")
    print(f"样本保存在: {samples_dir}")
    print("=" * 60)


if __name__ == "__main__":
    extract_test_samples()
