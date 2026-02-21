import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        """
        Multi-class Focal Loss supporting class-wise alpha and ignore_index.
        Args:
            gamma: focusing parameter
            alpha: class weights (list, tensor or float)
            ignore_index: target value to ignore
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = alpha  # scalar, binary classification
        else:
            self.alpha = None
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: [B, C, ...] logits
            target: [B, ...] int class labels
        """
        if input.size(0) != target.size(0):
            raise ValueError("Input and target batch size must match")

        num_classes = input.size(1)
        logpt = F.log_softmax(input, dim=1)
        pt = logpt.exp()

        # Gather log probs and probs corresponding to target labels
        target = target.long()
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        # Compute focal loss
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != input.device:
                    self.alpha = self.alpha.to(input.device)
                at = self.alpha[target]
            elif isinstance(self.alpha, float):  # binary classification
                at = torch.ones_like(target, dtype=torch.float32)
                at[target == 1] = self.alpha
                at[target == 0] = 1 - self.alpha
            else:
                raise TypeError("alpha must be float, list or tensor")
            loss = -at * focal_term * logpt
        else:
            loss = -focal_term * logpt

        # Mask ignore_index
        if self.ignore_index >= 0:
            ignore_mask = target == self.ignore_index
            loss = loss.masked_fill(ignore_mask, 0)
            valid_count = (~ignore_mask).sum()
        else:
            valid_count = loss.numel()

        # Reduction
        if self.reduction == 'mean':
            loss = loss.sum() / valid_count
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
