import torch 
import torch.nn.functional as F


def compute_ground_truth_boundaries(labels):
        """Compute binary boundary map from segmentation labels using Sobel-like edge detection"""
        if labels.dim() == 4:  # [B, n_class, H, W]
            labels = torch.argmax(labels, dim=1)  # [B, H, W]
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=labels.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=labels.device).view(1, 1, 3, 3)
        
        labels = labels.unsqueeze(1).float()  # [B, 1, H, W]
        edge_x = F.conv2d(labels, sobel_x, padding=1)
        edge_y = F.conv2d(labels, sobel_y, padding=1)
        boundary_map = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        return (boundary_map > 0.1).float() # 这里返回的是一个二值化的边界图

def boundary_loss( pred_edge, gt_edge):
        """Compute boundary loss between predicted and ground-truth boundary maps"""
        return F.binary_cross_entropy(pred_edge, gt_edge)

def get_boundary_loss(preds, labels):
    preds = F.softmax(preds, dim=1)
    preds_edge = compute_ground_truth_boundaries(preds)
    labels_edge = compute_ground_truth_boundaries(labels)
    loss = boundary_loss(preds_edge, labels_edge)
    return loss