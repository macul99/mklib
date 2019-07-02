import torch
from torch import nn, Tensor
import fastai
from . import pthutils
from . import pthlayer
import math
import numpy as np
import random


# create convolution layer based discriminator
def create_conv_discriminator(input_channel_num, bnEps:float=2e-5, bnMom:float=0.9, dropout=False, drop_p=0.15):
    num_layer = np.floor(math.log(input_channel_num,8)).astype(np.uint8)
    layer_channel = [input_channel_num]
    for i in range(num_layer-1):
        layer_channel.append(int(layer_channel[-1]/8))
    layer_channel.append(1) # last layer channel number set to 1

    middle_conv = []

    for i in range(len(layer_channel)-1):
        middle_conv.append(nn.Conv2d(layer_channel[i], layer_channel[i+1], kernel_size=3, stride=2, padding=1, bias=False))
        middle_conv.append(nn.BatchNorm2d(layer_channel[i+1], eps=bnEps, momentum=bnMom, affine=True))
        if i == len(layer_channel)-2:
            middle_conv.append(nn.ReLU())
        else:
            middle_conv.append(nn.PReLU(num_parameters=layer_channel[i+1]))
            if dropout:
                middle_conv.append(nn.Dropout2d(drop_p))
    middle_conv.append(pthlayer.Flatten())
    return nn.Sequential(*middle_conv)


class CriticModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        preModel = torch.load(opt.pretrainModel)
        preModel = list(preModel.children())[0]
        self.encoder = pthutils.cut_model(preModel,opt.ftExtractorCutNum)
        imsize = (112,112)
        sfs_szs = fastai.callbacks.hooks.model_sizes(self.encoder, size=imsize)
        input_ch_num = sfs_szs[-1][1]

        if opt.discriminator_type == 'pair':
            self.discriminator_pair = create_conv_discriminator(input_ch_num*2, bnEps=opt.bn_eps, bnMom=opt.bn_mom) # for conditional gan
            self.discriminator_unpair = None
        elif opt.discriminator_type == 'unpair':
            self.discriminator_pair = None
            self.discriminator_unpair = create_conv_discriminator(input_ch_num, bnEps=opt.bn_eps, bnMom=opt.bn_mom) # for unconditional gan
        else: # 'both'
            self.discriminator_pair = create_conv_discriminator(input_ch_num*2, bnEps=opt.bn_eps, bnMom=opt.bn_mom) # for conditional gan
            self.discriminator_unpair = create_conv_discriminator(input_ch_num, bnEps=opt.bn_eps, bnMom=opt.bn_mom) # for unconditional gan
        #fastai.torch_core.apply_init(self.model[2], nn.init.kaiming_normal_)

    def forward_pair(self, up_in:Tensor) -> Tensor:
        assert self.get_discriminator_pair is not None
        assert type(up_in)==type([]) and len(up_in)==2 # up_in[0] is for fake, up_in[1] is for real
        en1 = self.encoder(up_in[0])
        en2 = self.encoder(up_in[1])
        en = torch.cat((en1,en2),1)
        return self.discriminator_pair(en)

    def forward_unpair(self, up_in:Tensor) -> Tensor:
        assert self.get_discriminator_unpair is not None
        assert type(up_in)!=type([]) and type(up_in)==Tensor
        en = self.encoder(up_in)
        return self.discriminator_unpair(en)

    def forward(self, up_in:Tensor) -> Tensor:
        if self.opt.discriminator_type == 'pair':
            assert self.get_discriminator_pair is not None
            assert type(up_in)==type([]) and len(up_in)==2 # up_in[0] is for fake, up_in[1] is for real
            en1 = self.encoder(up_in[0])
            en2 = self.encoder(up_in[1])
            en = torch.cat((en1,en2),1)
            return self.discriminator_pair(en)
        elif self.opt.discriminator_type == 'unpair':
            assert self.get_discriminator_unpair is not None
            assert type(up_in)!=type([]) and type(up_in)==Tensor
            en = self.encoder(up_in)
            return self.discriminator_unpair(en)
        elif self.opt.discriminator_type == 'both':
            assert self.get_discriminator_pair is not None and self.get_discriminator_unpair is not None
            assert type(up_in)==type([]) and len(up_in)==3 # up_in[0] is for fake_pair, up_in[1] is for real_pair, up_in[2] is for unpair
            en1 = self.encoder(up_in[0])
            en2 = self.encoder(up_in[1])
            en3 = torch.cat((en1,en2),1)
            ft1 = self.discriminator_pair(en3)

            en4 = self.encoder(up_in[2])
            ft2 = self.discriminator_unpair(en4)
            return ft1, ft2
        else:
            raise NotImplementedError

    def get_discriminator_pair(self):
        return self.discriminator_pair

    def get_discriminator_unpair(self):
        return self.discriminator_unpair

    def freeze_encoder(self):
        pthutils.set_requires_grad(self.get_encoder(), False)

    def unfreeze_encoder(self):
        pthutils.set_requires_grad(self.get_encoder(), True)

    def freeze(self):
        pthutils.set_requires_grad(self.get_encoder(), False)
        pthutils.set_requires_grad(self.get_discriminator_pair(), False)
        pthutils.set_requires_grad(self.get_discriminator_unpair(), False)

    def unfreeze(self):
        pthutils.set_requires_grad(self.get_encoder(), True)
        pthutils.set_requires_grad(self.get_discriminator_pair(), True)
        pthutils.set_requires_grad(self.get_discriminator_unpair(), True)


class CriticModel_FIW(nn.Module): # customized for Family_in_Wild Kaggle challenge
    def __init__(self, opt):
        super().__init__()
        random.seed(100)
        self.opt = opt
        preModel = torch.load(opt.pretrainModel)
        preModel = list(preModel.children())[0]
        self.encoder = pthutils.cut_model(preModel,opt.ftExtractorCutNum)
        imsize = (112,112)
        sfs_szs = fastai.callbacks.hooks.model_sizes(self.encoder, size=imsize)
        input_ch_num = sfs_szs[-1][1]

        self.discriminator = create_conv_discriminator(input_ch_num*2, bnEps=opt.bn_eps, bnMom=opt.bn_mom) # for conditional gan
        #fastai.torch_core.apply_init(self.model[2], nn.init.kaiming_normal_)

    def forward(self, up_in:Tensor) -> Tensor:
        assert type(up_in)==type([]) and len(up_in)==3 # up_in[0] and up_in[1] are related to each other, bub not related to up_in[2]
        en1 = self.encoder(up_in[0])
        en2 = self.encoder(up_in[1])
        en3 = self.encoder(up_in[2])
        if random.random() > 0.5:
            en_pair = torch.cat((en1,en2),1)
        else:
            en_pair = torch.cat((en2,en1),1)

        if random.random() > 0.5:
            en_unpair1 = torch.cat((en1,en3),1)
        else:
            en_unpair1 = torch.cat((en3,en1),1)

        if random.random() > 0.5:
            en_unpair2 = torch.cat((en2,en3),1)
        else:
            en_unpair2 = torch.cat((en3,en2),1)
        
        fc1 = self.discriminator(en_pair)
        fc2 = self.discriminator(en_unpair1)
        fc3 = self.discriminator(en_unpair2)

        return fc1, fc2, fc3 


    def get_discriminator(self):
        return self.discriminator

    def freeze_encoder(self):
        pthutils.set_requires_grad(self.get_encoder(), False)

    def unfreeze_encoder(self):
        pthutils.set_requires_grad(self.get_encoder(), True)

    def freeze(self):
        pthutils.set_requires_grad(self.get_encoder(), False)
        pthutils.set_requires_grad(self.get_discriminator(), False)

    def unfreeze(self):
        pthutils.set_requires_grad(self.get_encoder(), True)
        pthutils.set_requires_grad(self.get_discriminator(), True)

'''
class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20,70,10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
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
'''