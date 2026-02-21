import os
import numpy as np
from PIL import Image

# CropSCD 标签颜色映射
cropscd_color_map = {
    0: (255, 255, 255),   # Unchanged
    1: (0, 112, 192),     # Water
    2: (0, 100, 0),       # Forest
    3: (128, 128, 0),     # Plantation
    4: (124, 252, 0),     # Grassland
    5: (112, 128, 144),   # Impervious Surface
    6: (200, 0, 0),       # Road
    7: (0, 255, 255),     # Greenhouse
    8: (244, 164, 96),    # Bare Soil
}

def index_to_rgb(label_img, color_map):
    """
    将索引标签图像转换为 RGB 图像
    :param label_img: 索引标签图像 (numpy array)
    :param color_map: 颜色映射字典
    :return: RGB 图像 (numpy array)
    """
    h, w = label_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        mask = label_img == idx
        rgb_img[mask] = color
    return rgb_img

def convert_labels(input_dir, output_dir, color_map):
    """
    转换指定文件夹下的所有标签图像
    :param input_dir: 输入标签文件夹路径
    :param output_dir: 输出 RGB 图像文件夹路径
    :param color_map: 颜色映射字典
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 读取索引标签图像
            input_path = os.path.join(input_dir, filename)
            label_img = np.array(Image.open(input_path).convert('L'))
            
            # 转换为 RGB 图像
            rgb_img = index_to_rgb(label_img, color_map)
            
            # 保存 RGB 图像
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(rgb_img).save(output_path)

if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = "/home/zyy/SCTDM_Net/results/CropSCD/test"
    output_dir = "/home/zyy/SCTDM_Net/results/CropSCD/label_rgb"
    
    # 执行转换
    convert_labels(input_dir, output_dir, cropscd_color_map)