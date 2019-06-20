import torch
import torch.nn as nn
from torch.nn import init
import functools
from collections import namedtuple
import math
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.callbacks  import hook_outputs
import torchvision.models as models
from ..pthnet import pthutils

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

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# GANLoss()
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# cal_gradient_penalty()
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# https://github.com/jantic/DeOldify/blob/master/fasterai/loss.py
# FeatureLoss()
class FeatureLoss_Vgg16(nn.Module):
    def __init__(self, device, layer_wgts=[20,70,10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.to(device).eval()
        requires_grad(self.m_feat, False)
        blocks = [i-1 for i,o in enumerate(children(self.m_feat)) if isinstance(o,nn.MaxPool2d)]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))] 
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input,target)]
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


class FeatureLoss_Resnet20(nn.Module):
    def __init__(self, device, layer_wgts=[20,70,10], model_path='/Projects/mk_utils/Convert_Mxnet_to_Pytorch/Pytorch_NewModel.pth'):
        super().__init__()

        preModel = torch.load(model_path)
        preModel = list(preModel.children())[0]
        self.m_feat = pthutils.cut_model(preModel,-3).to(device).eval()
        requires_grad(self.m_feat, False)
        layer_ids = [5,10,17]
        self.loss_features = [children(self.m_feat[5])[1][4], children(self.m_feat[10])[1][4], self.m_feat[17]]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))] 
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input,target)]
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

# https://github.com/fastai/fastai/blob/master/fastai/vision/gan.py
# AdaptiveLoss()
class AdaptiveLoss(nn.Module):
    "Expand the `target` to match the `output` size before applying `crit`."
    def __init__(self, crit):
        super().__init__()
        self.crit = crit

    def forward(self, output, target):
        return self.crit(output, target[:,None].expand_as(output).float())

# https://github.com/fastai/fastai/blob/master/fastai/vision/gan.py
# accuracy_thresh_expand()
def accuracy_thresh_expand(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy after expanding `y_true` to the size of `y_pred`."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true[:,None].expand_as(y_pred).byte()).float().mean()