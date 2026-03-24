import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
from typing import List, Optional, Literal
import math  # 新增：用于计算行数

def concat_figures(
    fig_list: List[plt.Figure],
    name_list: Optional[List[str]] = None,
    n_cols = 3,
    dpi: int = 100,
    background_color: str = 'white',
    close_originals: bool = True,
    font_path: Optional[str] = None,  # 字体路径，中文建议: 'simhei.ttf'
    font_size_ratio: float = 0.05     # 字体大小占图片高度的比例
) -> plt.Figure:
    """
    将多个 matplotlib Figure 对象拼接为一张大图，支持添加名称标签
    
    Args:
        fig_list: Figure 对象列表
        name_list: 与 fig_list 对应的名称列表（可选）
        direction: 拼接方向（当图片数量≤3时生效）
        dpi: 输出分辨率
        background_color: 背景颜色
        close_originals: 是否关闭原始 Figure 释放内存
        font_path: 字体文件路径（解决中文显示问题）
        font_size_ratio: 字体大小占图片高度的比例
    
    Returns:
        拼接后的 Figure 对象
    """
    # 过滤掉 None 和已关闭的 Figure
    valid_figs = [f for f in fig_list if f is not None and plt.fignum_exists(f.number)]
    if not valid_figs:
        raise ValueError("没有有效的 Figure 对象可供拼接")
    
    # 名称列表校验
    if name_list is not None:
        if len(name_list) != len(fig_list):
            raise ValueError(f"name_list 长度 ({len(name_list)}) 必须与 fig_list 长度 ({len(fig_list)}) 一致")
        valid_names = [name_list[i] for i, f in enumerate(fig_list) if f in valid_figs]
    else:
        valid_names = []
    
    # 转换为 PIL Image（创建独立副本）
    images = []
    for fig in valid_figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        images.append(img.copy())  # 复制脱离 buffer
        buf.close()
        
        if close_originals:
            plt.close(fig)
    
    # 添加名称标签
    if valid_names:
        labeled_images = []
        for img, name in zip(images, valid_names):
            # 计算文字区域高度（约为原图高度的 15%）
            text_height = int(img.height * 0.15)
            new_height = img.height + text_height
            
            # 创建新画布（原图 + 文字区域）
            new_img = Image.new('RGB', (img.width, new_height), background_color)
            new_img.paste(img, (0, 0))
            
            # 绘制文字
            draw = ImageDraw.Draw(new_img)
            font_size = int(img.height * font_size_ratio)
            
            # 加载字体（中文支持）
            try:
                if font_path and font_path.exists():
                    font = ImageFont.truetype(str(font_path), font_size)
                else:
                    font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # 计算居中位置
            bbox = draw.textbbox((0, 0), str(name), font=font)
            text_width = bbox[2] - bbox[0]
            x = (img.width - text_width) // 2
            y = img.height + (text_height - font_size) // 2
            
            # 绘制黑色文字
            draw.text((x, y), str(name), font=font, fill='black')
            
            labeled_images.append(new_img)
        images = labeled_images
    
    # 智能布局计算（核心修改部分）
    n_images = len(images)
    
    if n_images <= n_cols:
        # 图片数量≤n_cols时：垂直拼接
        widths, heights = zip(*(img.size for img in images))
        total_width, total_height = max(widths), sum(heights)
        coords = [(0, sum(heights[:i])) for i in range(n_images)]
    else:
        # 图片数量>n_cols时：网格布局（每行3列）
        cols = n_cols
        rows = math.ceil(n_images / cols)
        widths, heights = zip(*(img.size for img in images))
        
        # 计算每列的最大宽度和每行的最大高度
        col_max_widths = [0] * cols
        row_max_heights = [0] * rows
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            col_max_widths[col] = max(col_max_widths[col], img.width)
            row_max_heights[row] = max(row_max_heights[row], img.height)
        
        total_width = sum(col_max_widths)
        total_height = sum(row_max_heights)
        
        # 计算每个图片的坐标（按网格对齐）
        coords = []
        for idx in range(n_images):
            row = idx // cols
            col = idx % cols
            
            # 计算当前图片的左上角坐标
            x = sum(col_max_widths[:col])
            y = sum(row_max_heights[:row])
            coords.append((x, y))
    
    # 创建大图并拼接
    combined_image = Image.new('RGB', (total_width, total_height), background_color)
    for img, (x, y) in zip(images, coords):
        combined_image.paste(img, (x, y))
    
    # 转回 Figure
    fig, ax = plt.subplots(figsize=(total_width/dpi, total_height/dpi), dpi=dpi)
    ax.imshow(combined_image)
    ax.axis('off')
    
    return fig