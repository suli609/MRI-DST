import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')
#    focal_loss = FocalLoss(class_num=15,alpha=(0.40,0.20,0.20,0.20,0.20, 0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20),
#                           gamma=1, use_alpha=True, size_average=True)
#    loss = focal_loss(pred,label)
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def focal_loss    (pred,
                   target,
                   weight=None,
                   class_num = 15,
                #   alpha=0.75,
                   alpha=(0.1, 0.40,0.20,0.20,0.20,0.20, 0.20,0.20,0.20,0.20,0.20 ,0.20,0.20,0.20,0.20,0.20),
                   gamma=2.0,
                   use_alpha = False,
                   reduction='mean',
                   avg_factor=None):
    
    #统计该batch中各个类别的数目并赋值给alpha,总共target.size(0)个值
    cl_num = [0] * (class_num + 1)
    a = []
    use_log = True
    for j in range(0,target.size(0)):
        cl_num[target[j].item()] += 1
    
    for c in range(0,class_num + 1):
        if (cl_num[c] != 0) & (use_log):
            a.append(math.log( 1 + target.size(0)) / math.log( 1 + cl_num[c]))
        elif (cl_num[c] != 0) & (not use_log):
            a.append(target.size(0) / cl_num[c])
        else:
            a.append(0)
    for i in range(0,len(a)):
        a[i] = a[i] / sum(a)

    alpha = torch.tensor(a).cuda()
    
#    alpha = torch.tensor(alpha).cuda()
    # pred and target should be of the same size
    softmax = nn.Softmax(dim=1)

    prob = softmax(pred.view(-1,class_num + 1))
    prob = prob.clamp(min=0.00001,max=1.0)
    target_ = torch.zeros(target.size(0) , class_num + 1).cuda()
    target_.scatter_(1, target.view(-1, 1).long(), 1.)

#    print('pred size',pred.size()) #[2048,16]
#    print('prob size',prob.size()) #[2048,16]
#    print('prob size',prob[0][0]) #[2048,16]
#    print('target size',target.size()) #[2048]
#    print('target_ size',target_.size()) #[2048,16]
    assert prob.size() == target_.size()
    if  use_alpha:
    #    batch_loss = - alpha.double() * torch.pow(1-prob,gamma).double() * prob.log().double() * target_.double()
        batch_loss = - alpha.double() * torch.pow(1-prob,-1*prob.log()).double() * prob.log().double() * target_.double()
        batch_loss = batch_loss.sum(dim=1)
        loss = weight_reduce_loss(batch_loss, weight, reduction, avg_factor)
    else:
    #    batch_loss = - torch.pow(1-prob,gamma).double() * prob.log().double() * target_.double()
    #    batch_loss = - torch.pow(1-prob,-1*prob.log()).double() * prob.log().double() * target_.double()
        th = 0.25
        one_loss = torch.zeros_like(target).double()
        for i in range(0,target.size(0)):
            True_class = target[i].item()  #torch.item() 获取tensor的值
            prediction = prob[i][True_class]
            if prediction <= th :
                one_loss[i] = (- prob[i].log().double() * target_[i].double()).mean()
            #    print('one_class',one_loss[i])
            else:
                one_loss[i] = (- torch.pow((1-prob[i])/th,gamma).double()  * prob[i].log().double() * target_[i].double() ).mean()
            #    print('one_class',one_loss[i])
        one_loss = one_loss.mean()
        loss = weight_reduce_loss(one_loss, weight, reduction, avg_factor)


    return loss




def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 use_focal_loss=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.use_focal_loss = use_focal_loss
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_focal_loss:
            self.cls_criterion = focal_loss
        elif self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
            #self.cls_criterion = focal_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        
        if self.use_focal_loss:
            loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_num =15,
            alpha=(0.1, 0.40,0.20,0.20,0.20,0.20, 0.20,0.20,0.20,0.20,0.20 ,0.20,0.20,0.20,0.20,0.20),
            gamma=2.0,
            use_alpha = True,
        #    class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        else:
            loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
