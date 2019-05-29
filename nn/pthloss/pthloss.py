import torch
import torch.nn as nn
from torch.nn import init
import functools
from collections import namedtuple
import math

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels]

class PthArcLoss(nn.Module):
    def __init__(self, num_classes, emb_size, mrg_angle, mrg_scale, act_type='relu', grad_scale=1.0, easy_margin=False):
        assert(mrg_scale>0.0)
        assert(mrg_angle>=0.0)
        assert mrg_angle<(math.pi/2)

        super(ResnetBlock_V3, self).__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.mrg_angle = mrg_angle
        self.mrg_scale = mrg_scale
        self.act_type = act_type
        self.grad_scale = grad_scale
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.mrg_angle)
        self.sin_m = math.sin(self.mrg_angle)
        self.mm = math.sin(math.pi-self.mrg_angle)*self.mrg_angle
        self.threshold = math.cos(math.pi-self.mrg_angle)

        self.kernel = nn.Parameter(torch.Tensor(classnum, embedding_size))
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        if self.act_type == 'relu':
            self.act_1 = nn.ReLU(True)
        else:
            self.act_1 = nn.PReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, embbedings, label):
        label = label.view(-1)

        kernel_norm = l2_norm(self.kernel, axis=0)
        embeddings_norm = l2_norm(embeddings, axis=0)*self.mrg_scale
        fc7 = torch.mm(kernel_norm, embeddings_norm)

        zy = fc7.gather(1, label.view(-1,1))
        cos_t = zy/self.mrg_scale

        if self.easy_margin:
            cond = self.act_1(cos_t)
        else:
            cond_v = cos_t - self.threshold
            cond = self.act_1(cos_v)
        sin_t = torch.sqrt(1.0 - cos_t*cos_t)
        new_zy = cos_t*self.cos_m
        b = sin_t*self.sin_m
        new_zy = new_zy -b
        new_zy = new_zy*self.mrg_scale
        if self.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - self.mrg_scale*self.mm
        new_zy = new_zy.where(cond!=0, zy_keep)
        diff = new_zy - zy
        diff.unsqueeze_(1)
        gt_one_hot = one_hot_embedding(label, self.num_classes)
        body = gt_one_hot * diff

        fc7 = fc7 + body

        loss = self.loss(fc7, label)

        return self.grad_scale*loss


class PthRegressionLoss(nn.Module):
    def __init__(self, label_len, in_c, in_s=7, act_type='tanh', drop_rate=0.5, grad_scale=1.0):

        super(ResnetBlock_V3, self).__init__()
        self.label_len = label_len
        self.in_s = in_s
        self.in_c = in_c
        self.act_type = act_type
        self.drop_rate = drop_rate
        self.grad_scale = grad_scale
        self.model = [nn.AvgPool2D(self.in_s), nn.Dropout2d(p=self.drop_rate), nn.Linear(self.in_c, self.label_len, bias=True)]
        if self.act_type == 'tanh':
            self.model += [nn.Tanh()]
        else:
            assert False
        self.loss = nn.MSELoss()
        self.model = nn.Sequential(*self.model)

    def forward(self, data, label):
        label = label.view(-1)

        fc7 = self.model(data)

        loss = self.loss(fc7, label)

        return self.grad_scale*loss


class PthArcLandmarkLosses(nn.Module):
    def __init__(self, num_classes, emb_size, mrg_angle, mrg_scale, label_len, in_c, in_s, grad_scale=[1.0, 0.1]):
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.mrg_angle = mrg_angle
        self.mrg_scale = mrg_scale
        self.label_len = label_len
        self.in_c = in_c
        self.in_s = in_s
        self.grad_scale = grad_scale

        self.arcLoss = PthArcLoss(self.num_classes, self.emb_size, self.mrg_angle, self.mrg_scale, grad_scale=self.grad_scale[0])
        self.regLoss = PthRegressionLoss(self.label_len, self.in_c, self.in_s, grad_scale=self.grad_scale[1])

    def forward(self, pred, label, reg, lmk_gt):
        softmax_out = self.arcLoss(pred, label)
        lr_out = self.regLoss(reg, lmk_gt)
        return softmax_out, lr_out

