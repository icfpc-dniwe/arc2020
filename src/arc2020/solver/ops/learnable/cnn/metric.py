import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict


def multi_label_cross_entropy(preds: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor = None):
    preds = preds.reshape(preds.size(0), preds.size(1), -1)
    labels = labels.reshape(labels.size(0), -1)
    loss = F.cross_entropy(preds, labels, weight=weight)
    return loss


def multi_label_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    accuracy = np.mean(np.argmax(preds.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())
    return accuracy


def tag_accuracy(preds: torch.Tensor, tags: torch.Tensor) -> np.ndarray:
    thresh = 0.5
    return np.mean((preds.detach().cpu().numpy() > thresh) == tags.detach().cpu().numpy())


def masked_multi_label_accuracy(preds: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor):
    num_correct = np.sum(np.argmax(preds.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())
    masks = masks.detach().cpu().numpy()
    num_non_masked = np.sum(1 - masks)
    num_masked = np.sum(masks)
    accuracy = (num_correct - num_non_masked) / num_masked
    return accuracy


def weight_l2_norm(weights: Dict[str, torch.Tensor], multiplier: float = 1e-4) -> torch.Tensor:
    norm = 0
    for cur_weight in weights.values():
        norm = torch.mean(torch.sum(cur_weight ** 2, dim=1)) + norm
    return norm * multiplier


def size_loss(left_part: torch.Tensor, right_tensor: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(left_part, right_tensor)