import os
import json
import pandas as pd
import numpy as np
from PIL import Image

def get_v2_pallete(num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while lab > 0:
                pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
                pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
                pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    return v2_pallete


def overlay_semantic_images(base_image_path, overlay_image_path, COLOR_MAP_SS, base_alpha=0.3, overlay_alpha=0.7):
    base_image = Image.open(base_image_path).convert("RGBA")
    # overlay_image = Image.open(overlay_image_path).convert("RGBA")

    overlay_image_semantic = Image.open(overlay_image_path)
    overlay_image_semantic = np.array(overlay_image_semantic)
    overlay_image = np.zeros((np.array(overlay_image_semantic).shape[0], np.array(overlay_image_semantic).shape[1], 3), dtype=np.uint8)
    for i, rgb in zip(range(71), COLOR_MAP_SS):
        overlay_image[overlay_image_semantic == i] = rgb

    overlay_image = Image.fromarray(overlay_image.astype('uint8'))
    overlay_image = overlay_image.convert("RGBA")
    
    # 调整原图的alpha通道
    base_r, base_g, base_b, base_a = base_image.split()
    base_a = base_a.point(lambda p: p * base_alpha)
    base_image = Image.merge("RGBA", (base_r, base_g, base_b, base_a))
    
    # 调整掩码图像的alpha通道
    r, g, b, a = overlay_image.split()
    
    # 创建一个新的alpha通道，黑色部分设置为0，其他部分按alpha值调整
    new_alpha = a.point(lambda p: p * overlay_alpha)
    
    # 遍历每个像素，黑色部分的alpha设置为0
    width, height = overlay_image.size
    for x in range(width):
        for y in range(height):
            if r.getpixel((x, y)) == 0 and g.getpixel((x, y)) == 0 and b.getpixel((x, y)) == 0:
                new_alpha.putpixel((x, y), 0)
    
    # 合并处理后的alpha通道
    overlay_image = Image.merge("RGBA", (r, g, b, new_alpha))

    # 创建一个透明的图像，用于合成
    combined_image = Image.alpha_composite(base_image, overlay_image)
    return combined_image.convert("RGB")

def overlay_rgb_images(base_image_path, overlay_image_path, base_alpha=0.3, overlay_alpha=0.7):
    base_image = Image.open(base_image_path).convert("RGBA")
    overlay_image = Image.open(overlay_image_path).convert("RGBA")
    
    # 调整原图的alpha通道
    base_r, base_g, base_b, base_a = base_image.split()
    base_a = base_a.point(lambda p: p * base_alpha)
    base_image = Image.merge("RGBA", (base_r, base_g, base_b, base_a))
    
    # 调整掩码图像的alpha通道
    r, g, b, a = overlay_image.split()
    
    # 创建一个新的alpha通道，黑色部分设置为0，其他部分按alpha值调整
    new_alpha = a.point(lambda p: p * overlay_alpha)
    
    # 遍历每个像素，黑色部分的alpha设置为0
    width, height = overlay_image.size
    for x in range(width):
        for y in range(height):
            if r.getpixel((x, y)) == 0 and g.getpixel((x, y)) == 0 and b.getpixel((x, y)) == 0:
                new_alpha.putpixel((x, y), 0)
    
    # 合并处理后的alpha通道
    overlay_image = Image.merge("RGBA", (r, g, b, new_alpha))

    # 创建一个透明的图像，用于合成
    combined_image = Image.alpha_composite(base_image, overlay_image)
    return combined_image.convert("RGB")

    