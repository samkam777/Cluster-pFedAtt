import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os

class pFedMe_BCELoss(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(pFedMe_BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        #self.loss = nn.BCELoss()

    def forward(self, logits, target, reg_loss):
        #loss = - self.pos_weight * target * torch.log(logits) - \
        #    (1 - target) * torch.log(1 - logits)
        #print('loss before mean: {}\t'.format(loss))
        #print('reg_loss: {}\t'.format(reg_loss))
        
        #loss = self.loss(logits, target)
        #print('type of logits: {}\t'.format(type(logits)))
        #print('type of target: {}\t'.format(type(target)))
        #print('type of reg_loss: {}\t'.format(type(reg_loss)))
        loss = F.binary_cross_entropy(logits, target, reduction = "none")

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        #print('loss before mean: {}\t'.format(loss))
        loss += reg_loss
        #print('loss after mean: {}\t'.format(loss))
        return loss



