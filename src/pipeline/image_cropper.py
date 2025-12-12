"""
图像裁剪模块
从原始图像裁剪到作文正文区域
"""
from pathlib import Path
from PIL import Image


# 预定义裁剪区域 (从02_cropped_ocr_test.ipynb提取)
CROP_REGIONS = {
    'page1': {
        'x_start': 150,
        'y_start': 360,
        'x_end': 2320,
        'y_end': 3260
    },
    'page2': {
        'x_start': 150,
        'y_start': 50,
        'x_end': 2320,
        'y_end': 3150
    }
}


def crop_image(img_path, crop_region):
    """
    裁剪图片到指定区域
    
    参数:
        img_path: 图片路径 (str 或 Path)
        crop_region: 裁剪区域字典 {x_start, y_start, x_end, y_end}
    
    返回:
        PIL.Image: 裁剪后的图片
    """
    img = Image.open(img_path)
    cropped = img.crop((
        crop_region['x_start'],
        crop_region['y_start'],
        crop_region['x_end'],
        crop_region['y_end']
    ))
    return cropped


def auto_crop(img_path):
    """
    根据文件名自动识别并裁剪到正确区域
    
    参数:
        img_path: 图片路径 (str 或 Path)
    
    返回:
        PIL.Image: 裁剪后的图片
    
    异常:
        ValueError: 如果无法识别文件名格式或第2页区域未定义
    """
    filename = Path(img_path).name
    
    # 识别是第1页还是第2页
    if '_01.' in filename or '_01_' in filename:
        region = CROP_REGIONS['page1']
    elif '_02.' in filename or '_02_' in filename:
        region = CROP_REGIONS['page2']
        if region is None:
            raise ValueError("第2页裁剪区域尚未定义，请先完成Phase 2配置")
    else:
        raise ValueError(f"无法识别文件名格式: {filename}\n"
                        f"期望格式: sample_XX_01.png 或 sample_XX_02.png")
    
    return crop_image(img_path, region)


def get_crop_region_info():
    """
    获取当前配置的裁剪区域信息
    
    返回:
        dict: 包含各页裁剪区域的信息
    """
    info = {}
    for page, region in CROP_REGIONS.items():
        if region is not None:
            info[page] = {
                'defined': True,
                'coordinates': region,
                'size': (region['x_end'] - region['x_start'],
                        region['y_end'] - region['y_start'])
            }
        else:
            info[page] = {'defined': False}
    return info
