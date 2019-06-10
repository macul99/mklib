import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks.tensorboard import *
from fastai.vision.gan import *
from fastai.layers import *
from fastai.torch_core import *
from fastai.core import *
from fastai.callbacks  import hook_outputs
from fastai.callbacks.hooks import *
from . import pthutils
from . import pthlayer


class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, nf_factor:float=1.0, bnEps:float=2e-5, bnMom:float=0.9, **kwargs):
        super().__init__()
        self.hook = hook
        self.shuf = pthlayer.CustomPixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, bnEps=bnEps, bnMom=bnMom, **kwargs)
        self.bn = fastai.layers.batchnorm_2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = int((ni if final_div else ni//2)*nf_factor)
        self.conv1 = pthlayer.custom_conv_layer(ni, nf, leaky=leaky, bnEps=bnEps, bnMom=bnMom, **kwargs)
        self.conv2 = pthlayer.custom_conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, bnEps=bnEps, bnMom=bnMom, **kwargs)
        self.relu = fastai.layers.relu(leaky=leaky)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))

class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, n_out:int,  hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, bnEps:float=2e-5, bnMom:float=0.9, **kwargs):
        super().__init__()
        self.hook = hook
        up_out = x_out = n_out//2
        self.shuf = pthlayer.CustomPixelShuffle_ICNR(up_in_c, up_out, blur=blur, leaky=leaky, bnEps=bnEps, bnMom=bnMom, **kwargs)
        self.bn = fastai.layers.batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = pthlayer.custom_conv_layer(ni, x_out, leaky=leaky, self_attention=self_attention, bnEps=bnEps, bnMom=bnMom, **kwargs)
        self.relu = fastai.layers.relu(leaky=leaky)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)

class DynamicUnetDeep(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, self_attention:bool=False,
                 y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True, bottle:bool=False,
                 norm_type:Optional[fastai.vision.NormType]=fastai.vision.NormType.Batch, nf_factor:float=1.0, 
                 bnEps:float=2e-5, bnMom:float=0.9, hook_detach:bool=False, **kwargs):
        extra_bn =  norm_type == fastai.vision.NormType.Spectral
        imsize = (112,112)
        sfs_szs = fastai.callbacks.hooks.model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(pthutil.get_sfs_idxs(sfs_szs)))
        self.sfs = fastai.callbacks.hooks.hook_outputs([encoder[i] for i in sfs_idxs], detach=hook_detach)
        x = fastai.callbacks.hooks.dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(pthlayer.custom_conv_layer(ni, ni*2, norm_type=norm_type, extra_bn=extra_bn, bnEps=bnEps, bnMom=bnMom, **kwargs),
                                    pthlayer.custom_conv_layer(ni*2, ni, norm_type=norm_type, extra_bn=extra_bn, bnEps=bnEps, bnMom=bnMom, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)
            unet_block = UnetBlockDeep(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=blur, self_attention=sa,
                                   norm_type=norm_type, extra_bn=extra_bn, nf_factor=nf_factor, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(fastai.layers.PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(fastai.layers.MergeLayer(dense=True))
            ni += fastai.torch_core.in_channels(encoder)
            layers.append(fastai.layers.res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [pthlayer.custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)]
        if y_range is not None: layers.append(fastai.layers.SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

class DynamicUnetWide(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, self_attention:bool=False,
                 y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True, bottle:bool=False,
                 norm_type:Optional[fastai.vision.NormType]=fastai.vision.NormType.Batch, nf_factor:int=1,
                 bnEps:float=2e-5, bnMom:float=0.9, hook_detach:bool=False, **kwargs):
        
        nf = 512 * nf_factor
        extra_bn =  norm_type == fastai.vision.NormType.Spectral
        imsize = (112,112)
        sfs_szs = fastai.callbacks.hooks.model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(pthutils.get_sfs_idxs(sfs_szs)))
        self.sfs = fastai.callbacks.hooks.hook_outputs([encoder[i] for i in sfs_idxs], detach=hook_detach)
        x = fastai.callbacks.hooks.dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(pthlayer.custom_conv_layer(ni, ni*2, norm_type=norm_type, extra_bn=extra_bn, bnEps=bnEps, bnMom=bnMom, **kwargs),
                                    pthlayer.custom_conv_layer(ni*2, ni, norm_type=norm_type, extra_bn=extra_bn, bnEps=bnEps, bnMom=bnMom, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)

            n_out = nf if not_final else nf//2

            unet_block = UnetBlockWide(up_in_c, x_in_c, n_out, self.sfs[i], final_div=not_final, blur=blur, self_attention=sa,
                                   norm_type=norm_type, extra_bn=extra_bn, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(fastai.layers.PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(fastai.layers.MergeLayer(dense=True))
            ni += fastai.torch_core.in_channels(encoder)
            layers.append(fastai.layers.res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [pthlayer.custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)]
        if y_range is not None: layers.append(fastai.layers.SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()


class UnetWideModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        preModel = torch.load(opt.pretrainModel)
        preModel = list(preModel.children())[0]
        encoder = pthutils.cut_model(preModel,opt.ftExtractorCutNum)
        self.model = DynamicUnetWide(   encoder, n_classes=opt.num_classes_gen, blur=opt.blur, blur_final=opt.blur_final,
                                        self_attention=opt.self_attention, y_range=opt.y_range, norm_type=opt.norm_type_gen, 
                                        last_cross=opt.last_cross, bottle=opt.bottle, nf_factor=opt.nf_factor, bnEps=opt.bn_eps, 
                                        bnMom=opt.bn_mom, hook_detach=opt.hook_detach )

    def forward(self, up_in:Tensor) -> Tensor:
        return self.model(up_in)

    def get_encoder(self):
        return list(self.model.children())[0]

    def freeze_encoder(self):
        pthutils.set_requires_grad(self.get_encoder(), False)

    def unfreeze_encoder(self):
        pthutils.set_requires_grad(self.get_encoder(), True)

    def freeze(self):
        pthutils.set_requires_grad(self.model, False)

    def unfreeze(self):
        pthutils.set_requires_grad(self.model, True)


class FeatureExtractorModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.model = torch.load(opt.pretrainModel)

        # check feature_dim
        imsize = (112,112)
        sfs_szs = fastai.callbacks.hooks.model_sizes(self.model, size=imsize)
        assert opt.ftDim == sfs_szs[-1][1]

    def forward(self, up_in:Tensor) -> Tensor:
        return self.model(up_in)

    def freeze(self):
        pthutils.set_requires_grad(self.model, False)

    def unfreeze(self):
        pthutils.set_requires_grad(self.model, True)

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