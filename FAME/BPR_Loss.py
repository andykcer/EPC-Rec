import torch
import torch.nn.functional as F

def bpr_loss(pos_scores, neg_scores):
    """
    pos_scores: [B], 负样本个数为N
    neg_scores: [B, N]
    """
    loss = - (pos_scores.unsqueeze(1) - neg_scores).sigmoid().log().sum(dim=1).mean()
    return loss

