"""
PDF转图片工具
将作文PDF转换为单页图片,支持两页合并为一份作文
"""

try:
    import fitz  # PyMuPDF旧版本
except ImportError:
    import pymupdf as fitz  # PyMuPDF新版本
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np


class PDFConverter:
    """PDF转图片转换器"""
    
    def __init__(self, dpi: int = 300, merge_pages: bool = True):
        """
        初始化转换器
        
        Args:
            dpi: 输出图片DPI,默认300(高质量)
            merge_pages: 是否每两页合并为一张图片(针对两页一篇作文的情况)
        """
        self.dpi = dpi
        self.merge_pages = merge_pages
        self.zoom = dpi / 72  # PDF默认72 DPI
        
    def convert_pdf(
        self, 
        pdf_path: str, 
        output_dir: str,
        prefix: str = "page"
    ) -> List[str]:
        """
        转换单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            prefix: 输出文件名前缀
            
        Returns:
            生成的图片文件路径列表
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        output_files = []
        
        print(f"转换 {pdf_path.name}: {total_pages} 页")
        
        if self.merge_pages:
            # 每两页合并为一张图片
            for i in range(0, total_pages, 2):
                if i + 1 < total_pages:
                    # 合并两页
                    img_path = self._merge_two_pages(
                        doc, i, i+1, output_dir, f"{prefix}_{i//2+1:03d}"
                    )
                else:
                    # 只有一页
                    img_path = self._save_single_page(
                        doc, i, output_dir, f"{prefix}_{i//2+1:03d}"
                    )
                output_files.append(str(img_path))
        else:
            # 每页单独保存
            for i in range(total_pages):
                img_path = self._save_single_page(
                    doc, i, output_dir, f"{prefix}_{i+1:03d}"
                )
                output_files.append(str(img_path))
        
        doc.close()
        return output_files
    
    def _save_single_page(
        self, 
        doc: fitz.Document, 
        page_num: int, 
        output_dir: Path, 
        filename: str
    ) -> Path:
        """保存单个页面为图片"""
        page = doc.load_page(page_num)
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        output_path = output_dir / f"{filename}.png"
        pix.save(str(output_path))
        
        return output_path
    
    def _merge_two_pages(
        self, 
        doc: fitz.Document, 
        page1_num: int, 
        page2_num: int, 
        output_dir: Path, 
        filename: str
    ) -> Path:
        """合并两页为一张图片(上下拼接)"""
        # 获取两页的图片
        mat = fitz.Matrix(self.zoom, self.zoom)
        
        page1 = doc.load_page(page1_num)
        pix1 = page1.get_pixmap(matrix=mat, alpha=False)
        img1 = Image.frombytes("RGB", [pix1.width, pix1.height], pix1.samples)
        
        page2 = doc.load_page(page2_num)
        pix2 = page2.get_pixmap(matrix=mat, alpha=False)
        img2 = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)
        
        # 上下拼接
        total_height = img1.height + img2.height
        max_width = max(img1.width, img2.width)
        
        merged = Image.new('RGB', (max_width, total_height), 'white')
        merged.paste(img1, (0, 0))
        merged.paste(img2, (0, img1.height))
        
        output_path = output_dir / f"{filename}.png"
        merged.save(str(output_path))
        
        return output_path
    
    def batch_convert(
        self, 
        pdf_dir: str, 
        output_base_dir: str,
        pattern: str = "*.pdf"
    ) -> dict:
        """
        批量转换文件夹中的所有PDF
        
        Args:
            pdf_dir: PDF文件夹路径
            output_base_dir: 输出基础目录
            pattern: 文件匹配模式
            
        Returns:
            转换结果字典: {pdf_name: [image_paths]}
        """
        pdf_dir = Path(pdf_dir)
        output_base_dir = Path(output_base_dir)
        
        results = {}
        pdf_files = sorted(pdf_dir.glob(pattern))
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_file in pdf_files:
            # 为每个PDF创建独立的输出目录
            pdf_name = pdf_file.stem
            output_dir = output_base_dir / pdf_name
            
            try:
                image_paths = self.convert_pdf(
                    str(pdf_file), 
                    str(output_dir),
                    prefix=pdf_name
                )
                results[pdf_name] = image_paths
                print(f"✓ {pdf_name}: 生成 {len(image_paths)} 张图片")
            except Exception as e:
                print(f"✗ {pdf_name}: 转换失败 - {e}")
                results[pdf_name] = []
        
        return results


def main():
    """命令行入口示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF转图片工具")
    parser.add_argument("pdf_path", help="PDF文件或文件夹路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--dpi", type=int, default=300, help="输出DPI(默认300)")
    parser.add_argument("--no-merge", action="store_true", help="不合并两页")
    
    args = parser.parse_args()
    
    converter = PDFConverter(dpi=args.dpi, merge_pages=not args.no_merge)
    
    pdf_path = Path(args.pdf_path)
    if pdf_path.is_file():
        # 单个文件
        converter.convert_pdf(str(pdf_path), args.output)
    elif pdf_path.is_dir():
        # 批量转换
        converter.batch_convert(str(pdf_path), args.output)
    else:
        print(f"错误: {pdf_path} 不存在")


if __name__ == "__main__":
    main()
