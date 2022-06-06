import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    crossentropy_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)  # N*1*H*W => N*H*W
    if input.shape[-1] != target.shape[-1]:  # match the size
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)


def weighted_cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    cross_entropy_loss with weight
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)  # N*1*H*W => N*H*W
    if input.shape[-1] != target.shape[-1]:  # match the size
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    nonbd = torch.sum(1 - target) / torch.sum(torch.ones_like(target))
    weight = torch.FloatTensor([1 - nonbd, nonbd]).cuda()

    return F.cross_entropy(input=input, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)


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
    def __init__(self, smooth=1e-5, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - 2 * num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class MixLoss(nn.Module):
    """
        alpha * BCELoss + beta * DiceLoss
    """
    def __init__(self, alpha=1, beta=1):
        super(MixLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.diceloss = BinaryDiceLoss()

    def forward(self, pred, target, weight=None):
        # Dice Loss
        prediction = F.softmax(pred, dim=1)
        # prediction = prediction.max(1)[0]
        prediction = prediction[:, 1, :, :]
        # prediction = torch.argmax(prediction, dim=1)
        dice_loss = self.diceloss(prediction, target)

        # Binary CrossEntropy
        # bce_loss = nn.BCELoss()
        #bce_loss = F.binary_cross_entropy(prediction, target)
        bce_loss = F.cross_entropy(input=pred, target=target, reduction='mean', weight=weight)

        mix_loss = self.alpha * bce_loss + self.beta * dice_loss

        return mix_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, **kwargs):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)  # N,H,W => N*H*W,1

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)  # 取出真实标签对应的预测概率
        logpt = logpt.view(-1)  # N*H*W
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))  # 取出真实标签对应的权重
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def boundary_loss(input, target, **kwargs):
    """
    input : B,H,W
    target : B,H,W
    """
    input = input.float()
    target = target.float()
    pool_size = 7  # control the width of the boundary
    maxpool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
    target = torch.abs(target - maxpool(target))
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)  # N*1*H*W => N*H*W
    if input.shape[-1] != target.shape[-1]:  # match the size
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    pred = input
    nonbd = torch.sum(1 - target, dim=(1, 2), keepdim=True) / torch.sum(torch.ones_like(target),
                                                                        dim=(1, 2), keepdim=True)  # weights (size:B,1,1)
    pred = F.sigmoid(pred)  # size:B,H,W
    # pred = torch.abs(pred - maxpool(pred))
    loss = (-1) * torch.mean(nonbd * torch.log(pred + 1e-7) * target + (1 - nonbd) * torch.log(1 - pred + 1e-7) *
                             (1 - target), dim=(1, 2))  # size:B
    # for i in range(num):
    #     pred = torch.unsqueeze(input[i], dim=0)  # size:1,H,W
    #     gt = torch.unsqueeze(target[i], dim=0)
    #     nonbd = torch.sum(1 - target[i]) / torch.sum(torch.ones_like(target[i]))
    #     pred = F.sigmoid(pred)
    #     tmp = (-1) * torch.mean(nonbd * torch.log(pred) * gt + (1 - nonbd) * torch.log(1 - pred) * (1 - gt))
    #     loss += tmp
    return torch.mean(loss)


class BDEnhancedCELoss(nn.Module):
    """
        BDEnhancedCELoss : alpha * BCELoss + beta * BoundaryLoss
    """
    def __init__(self, alpha=1, beta=1):
        super(BDEnhancedCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pool_size = 7  # size of neighborhood
        self.avgpool = nn.AvgPool2d(self.pool_size, stride=1, padding=self.pool_size // 2)
        # self.maxpool = nn.MaxPool2d(self.pool_size, stride=1, padding=self.pool_size // 2)

    def forward(self, pred, target, weight=None):
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)  # N*1*H*W => N*H*W
        if pred.shape[-1] != target.shape[-1]:  # match the size
            pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=True)

        # Binary CrossEntropy Loss
        bce_loss = F.cross_entropy(input=pred, target=target, reduction='mean', weight=weight)

        # Boundary Loss
        target = target.float()  # avgpool2d cannot compute the tensor whose type is 'Long'
        boundary = torch.abs(self.avgpool(target) - target)  # weight matrix, size : B,H,W
        # boundary = torch.abs(self.maxpool(target) - target)  # weight matrix, size : B,H,W
        prob = F.softmax(pred, dim=1)  # size: B,2,H,W
        # print(torch.min(prob[:, 1, :, :]))
        prob = torch.log(prob + 1e-7)
        # print(torch.max((-1) * prob[:, 1, :, :]))
        bd_loss = torch.mean((-1) * boundary * prob[:, 1, :, :] * target - boundary * prob[:, 0, :, :] * (1 - target))
        # bd_loss = torch.mean((-1) * boundary * prob[:, 1, :, :] - (1 - boundary) * prob[:, 0, :, :])

        self.loss = self.alpha * bce_loss + self.beta * bd_loss

        return self.loss
