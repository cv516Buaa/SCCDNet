import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

@MODELS.register_module()
class FRCLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=10,
                 loss_name='loss_contrastive'):
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
                feat,
                label,
                weight=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""

        label[label == 255] = 0

        assert feat.shape[0] == label.shape[0]
        assert feat.shape[2] == label.shape[2]
        assert feat.shape[3] == label.shape[3]

        embedding_list = torch.zeros([self.num_classes, feat.shape[1]], dtype=torch.float, device='cuda:0')
        num_list = torch.zeros([self.num_classes], dtype=torch.float, device='cuda:0')
        for i in range(label.shape[0]):
            mask = label[i]
            while(mask.max() > 0):
                cls = mask.max()
                mask_tem = torch.where(mask == cls, 1, 0)
                feat_tem = feat[i] * mask_tem
                embedding_list[cls.int(), :] += feat_tem.mean(2).mean(1) * label.shape[2] * label.shape[3] / (mask_tem.sum() + 1e-6)
                num_list[cls.int()] += 1
                mask = torch.where(mask == cls, 0, mask)
        embeddings = []
        for i in range(embedding_list.shape[0]):
            if num_list[i] != 0:
                embeddings.append(embedding_list[i, :].unsqueeze(0) / num_list[i])
        embeddings = torch.cat(embeddings, dim=0)
        embed1 = embeddings.unsqueeze(0)
        embed2 = embeddings.unsqueeze(1)

        cos_matrix = (embed1 * embed2)
        cos_matrix = cos_matrix.sum(2) / ((embed1 ** 2).sum(2) * (embed2 ** 2).sum(2))
        target = torch.arange(0, cos_matrix.shape[0], device=cos_matrix.device)
        loss_cls = F.cross_entropy(cos_matrix, target)

        return loss_cls * self.loss_weight

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
