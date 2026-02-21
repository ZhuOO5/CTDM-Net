import torch
import numpy as np
import os
from metrics import eval_metrics_direct, get_seg_metrics  # 确保 metrics.py 里有这些函数
from PIL import Image  # 处理 PNG 图片
import argparse  # 用于解析命令行参数

def load_data(test_file):
    """ 读取 test.txt，获取测试样本的文件名列表 """
    with open(test_file, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def load_predictions(prediction_file):
    """ 加载预测结果（PNG 格式），转换为 PyTorch Tensor """
    try:
        pred = Image.open(prediction_file).convert('L')  # 转换为灰度图，确保单通道
        pred = np.array(pred)  # 转换为 NumPy 数组
        pred = torch.from_numpy(pred).long()  # 转换为 PyTorch Tensor
        return pred
    except Exception as e:
        print(f"Error loading prediction file {prediction_file}: {e}")
        return None  # 发生错误时返回 None

def load_labels(label_dir, test_data):
    """ 加载真实标签数据，并转换为 PyTorch Tensor """
    labels = []
    for file_name in test_data:
        label_file = os.path.join(label_dir, os.path.basename(file_name))
        try:
            label = Image.open(label_file).convert('L')  # 确保单通道
            label = np.array(label)
            label_tensor = torch.from_numpy(label).long()
            labels.append(label_tensor)
        except Exception as e:
            print(f"Error loading label file {label_file}: {e}")
            labels.append(None)  # 发生错误时加入 None
    return labels

def save_evaluation_results(metrics, output_dir):
    """ 保存评估结果到 evaluation.txt """
    evaluation_file = os.path.join(output_dir, 'evaluation.txt')
    with open(evaluation_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def main(test_file, prediction_dir, label_dir, num_classes):
    """ 主函数，执行评估 """
    # 加载测试数据的文件名列表
    test_data = load_data(test_file)

    # 按 test.txt 里的顺序加载预测结果
    predictions = [load_predictions(os.path.join(prediction_dir, f)) for f in test_data]
    labels = load_labels(label_dir, test_data)

    total_correct = 0
    total_labeled = 0
    total_inter = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_FPR = np.zeros(num_classes)
    total_pred = np.zeros(num_classes)

    for i, target in enumerate(labels):
        if predictions[i] is None or target is None:
            print(f"Skipping sample {test_data[i]} due to missing data.")
            continue

        correct, labeled, inter, union, area_FPR, area_pred = eval_metrics_direct(predictions[i], target, num_classes)
        total_correct += correct
        total_labeled += labeled
        total_inter += inter
        total_union += union
        total_FPR += area_FPR
        total_pred += area_pred

    # 计算最终的评估指标
    metrics = get_seg_metrics(total_correct, total_labeled, total_inter, total_union, total_FPR, total_pred, num_classes)

    # 保存评估结果
    # output_dir = os.path.dirname(prediction_dir)  # 结果保存到与 prediction_dir 同级的目录
    eval_dir = os.path.join(prediction_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    save_evaluation_results(metrics, eval_dir)

    print("Evaluation results saved to evaluation.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--test_file', type=str, default='/home/zyy/Semantic_change_detection/dataset/JL1_2023/reorganized/test.txt', help='Path to the test file')
    parser.add_argument('--prediction_dir', type=str, default='results/JL1-2023/MeGNet/test2', help='Directory containing prediction files')
    parser.add_argument('--label_dir', type=str, default='/home/zyy/Semantic_change_detection/dataset/JL1_2023/reorganized/label', help='Directory containing label files')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
    args = parser.parse_args()
    main(args.test_file, args.prediction_dir, args.label_dir, args.num_classes)
