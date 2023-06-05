import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

@MODELS.register_module()
class DistillLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=10,
                 loss_name='loss_distill'):
        super().__init__()
        assert (use_sigmoid is False)
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                pred,
                slabel,
                weight=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        # assert reduction_override in (None, 'none', 'mean', 'sum')

        loss = (pred - slabel) ** 2
        loss = loss.mean(1).mean(1).mean(1)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name