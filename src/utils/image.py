from typing import List, Optional, Tuple
import math
from PIL import Image, ImageDraw, ImageFont


def stitch_images(
    images: List[Image.Image],
    n_cols: int = 3,
    margin: int = 5,
    keep_border: bool = False,
    texts: Optional[List[str]] = None,
) -> Image.Image:
    """ 将一组 PIL.Image.Image 按左到右、从上到下的顺序拼接成网格图像。
        - images: 图像列表，数量任意（不足的格子将使用白色填充）
        - n_cols: 每行的列数（网格的列数）
        - margin: 图像之间的边距（像素）
        - output_path: 拼接后保存的文件路径
        - keep_border: 是否保留拼接图像外围边距；默认为 False
        - texts: 每个网格对应的文本标注（长度可变）。若提供文本，将在相应网格的左上角绘制，背景为半透明黑色，文字为白色
    """
    if n_cols < 1:
        raise ValueError("n_cols 必须大于等于 1")
    
    m = len(images)
    if m == 0:
        raise ValueError("需要至少一张图像")
    
    # 统一到 RGB，便于后续处理
    imgs = [img.convert('RGB') for img in images]

    # 网格单元尺寸取所有图片的最大宽高
    max_w = max(img.width for img in imgs)
    max_h = max(img.height for img in imgs)

    n_rows = math.ceil(m / n_cols)
    cell_w, cell_h = max_w, max_h

    # 计算画布尺寸（包含四周和格间 margin）
    total_w = n_cols * cell_w + margin * (n_cols + 1)
    total_h = n_rows * cell_h + margin * (n_rows + 1)

    # 使用 RGBA 以便后续叠加半透明文本底
    stitched = Image.new('RGBA', (total_w, total_h), (255, 255, 255, 255))

    # 粘贴每张图片
    for idx, img in enumerate(imgs):
        row = idx // n_cols
        col = idx % n_cols
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        
        if img.size != (cell_w, cell_h):
            src_w, src_h = img.size
            dst_w, dst_h = (int(src_w * cell_h / src_h), cell_h) if cell_h < cell_w else (cell_w, int(src_h * cell_w / src_w))
            img_resized = img.resize((dst_w, dst_h), resample=Image.LANCZOS)
            if img_resized.size != (cell_w, cell_h):
                # 居中粘贴
                paste_x = x + (cell_w - dst_w) // 2
                paste_y = y + (cell_h - dst_h) // 2
                x, y = paste_x, paste_y
        else:
            img_resized = img

        img_rgba = img_resized.convert('RGBA')
        stitched.paste(img_rgba, (x, y), img_rgba)

    # 绘制文本（半透明黑底白字，位于每格左上角）
    if texts:
        font_size = int(cell_h / 8)
        font = ImageFont.load_default(size=font_size)

        overlay = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')

        for idx in range(min(len(texts), m)):
            text = texts[idx]
            if not text:
                continue

            row = idx // n_cols
            col = idx % n_cols
            x = margin + col * (cell_w + margin)
            y = margin + row * (cell_h + margin)

            # 计算文本尺寸
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(text, font=font)

            padding = int(font_size / 8)
            box_w = int(1.05 * text_w) + 2 * padding
            box_h = int(1.3 * font_size) + 2 * padding

            # 初始定位在格子内左上角，带内边距
            left = x
            top = y
            right = left + box_w
            bottom = top + box_h

            # 半透明黑底
            draw.rectangle([(left, top), (right, bottom)], fill=(0, 0, 0, 128))
            # 白字
            draw.text((left + padding, top + padding), text, fill=(255, 255, 255, 255), font=font)

        stitched = Image.alpha_composite(stitched, overlay)

    # 转回 RGB 输出
    final = stitched.convert('RGB')

    # 可选裁剪外边距
    if not keep_border:
        left = margin
        top = margin
        right = total_w - margin
        bottom = total_h - margin
        final = final.crop((left, top, right, bottom))

    return final
