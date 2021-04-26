#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        ## TODO: here see if torch or numpy is faster
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss

class ECELoss(nn.Module):
    def __init__(self, thresh=0.0, n_min=0.0, n_classes=19, alpha=1, radius=1, beta=0.1, ignore_lb=255):
        super(ECELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.n_classes = n_classes
        self.alpha = alpha
        self.radius = radius
        self.beta = beta

        self.criteria = nn.BCELoss()
        self.edge_criteria = EdgeLoss(self.n_classes, self.radius, self.alpha)


    def forward(self, logits, labels):

        labels_onehot = F.one_hot(labels, 2)
        labels_onehot = labels_onehot.permute([0, 3, 1, 2])

        if self.beta > 0:
            return self.criteria(logits, labels_onehot.float()) + self.beta*self.edge_criteria(logits, labels)
        else:
            return self.criteria(logits, labels)

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=0.01):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha

        self.criteria = nn.BCELoss()


    def forward(self, logits, label):

        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        filt2 = torch.ones(self.n_classes, 1, ks, ks)
        filt2[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt2.requires_grad = False
        filt2 = filt2.cuda()
        prediction = logits
        prededge = F.conv2d(prediction.float(), filt2, bias=None, stride=1, padding=self.radius, groups=self.n_classes)
        norm = torch.sum(torch.pow(prededge,2), 1).unsqueeze(1)
        prededge = norm/(norm + self.alpha)
        return BinaryDiceLoss()(prededge.float(), lbedge.float())

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.sum()/target.size(0)