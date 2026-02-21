import os
import numpy as np
from PIL import Image

# JL1 label to RGB
jl1_color_map = {
    0: (255, 255, 255),    # No change
    1: (200, 0, 0),        # Farmland → Road
    2: (0, 100, 0),        # Farmland → Trees
    3: (112, 128, 144),    # Farmland → Buildings
    4: (160, 82, 45),      # Farmland → Others
    5: (255, 160, 160),    # Road → Farmland
    6: (144, 238, 144),    # Trees → Farmland
    7: (200, 200, 200),    # Buildings → Farmland
    8: (210, 180, 140),    # Others → Farmland
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
        if filename.endswith(('.png', '.jpg', '.jpeg','.tif')):
            # 读取索引标签图像
            input_path = os.path.join(input_dir, filename)
            label_img = np.array(Image.open(input_path).convert('L'))
            
            # 转换为 RGB 图像
            rgb_img = index_to_rgb(label_img, color_map)
            
            # 提取文件名（不含扩展名）
            name_without_ext = os.path.splitext(filename)[0]
            # 构造 PNG 格式的输出路径
            output_path = os.path.join(output_dir, f"{name_without_ext}.png")
            Image.fromarray(rgb_img).save(output_path, "PNG")

if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = "results/JL1_2023/vision_test2"
    output_dir = "results/JL1_2023/vision_test2_rgb"
    
    # 执行转换
    convert_labels(input_dir, output_dir, jl1_color_map)