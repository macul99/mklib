import fastai
from fastai.core import *
from torch import nn


def custom_conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, is_1d:bool=False,
               norm_type:Optional[fastai.vision.NormType]=fastai.vision.NormType.Batch,  bnEps:float=2e-5, bnMom:float=0.9, use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False,
               extra_bn:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (fastai.vision.NormType.Batch, fastai.vision.NormType.BatchZero) or extra_bn==True
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = fastai.torch_core.init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==fastai.vision.NormType.Weight:   conv = nn.utils.weight_norm(conv)
    elif norm_type==fastai.vision.NormType.Spectral: conv = nn.utils.spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(fastai.layers.relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf, eps=bnEps, momentum=bnMom))
    if self_attention: layers.append(fastai.layers.SelfAttention(nf))
    return nn.Sequential(*layers)

class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=False, leaky:float=None, bnEps:float=2e-5, bnMom:float=0.9, **kwargs):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(ni, nf*(scale**2), ks=1, use_activ=False, bnEps=bnEps, bnMom=bnMom, **kwargs)
        fastai.layers.icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = fastai.layers.relu(True, leaky=leaky)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x
