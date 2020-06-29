import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class BinaryCrossEntropy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, inputs, mask, smooth=1):

        inputs = torch.sigmoid(inputs)
        entropy_weight = 0.5
        inputs = inputs.view(-1)
        mask = mask.view(-1)
        intersection = (inputs * mask).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + mask.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, mask, reduction='mean')
        loss_final = BCE * entropy_weight + dice_loss * (1 - entropy_weight)
        return loss_final
