import torch
import torch.nn as nn
from torch.nn import init
import functools
from . import pthlayer


class ResnetBlock_V2(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_c, K, stride, act_type='prelu', red=False, bnEps=2e-5, bnMom=0.9, bottle_neck=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_V2, self).__init__()
        self.in_c = in_c
        self.K = K
        self.stride = stride
        self.act_type = act_type
        self.red = red
        self.bnEps = bnEps
        self.bnMom = bnMom
        self.bottle_neck = bottle_neck
        self.bn_stem = nn.BatchNorm2d(self.in_c, eps=self.bnEps, momentum=self.bnMom, affine=True)
        if act_type == 'prelu':
            self.act_stem = nn.PReLU(self.in_c)
        else:
            self.act_stem = nn.ReLU(True)
        if self.red:
            self.conv_red = nn.Conv2d(in_c, K, kernel_size=1, stride=self.stride, padding=0, bias=False)
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []

        if self.bottle_neck:
            if self.act_type == 'prelu':
                conv_block += [ nn.Conv2d(self.K, int(self.K*0.25), kernel_size=1, padding=0, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.PReLU(int(self.K*0.25))]

                conv_block += [ nn.Conv2d(int(self.K*0.25), int(self.K*0.25), kernel_size=3, stride=self.stride, padding=1, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.PReLU(int(self.K*0.25)),
                                nn.Conv2d(int(self.K*0.25), self.K, kernel_size=1, stride=1, padding=0, bias=False)]
            else:
                conv_block += [ nn.Conv2d(self.K, int(self.K*0.25), kernel_size=1, padding=0, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.ReLU()]

                conv_block += [ nn.Conv2d(int(self.K*0.25), int(self.K*0.25), kernel_size=3, stride=self.stride, padding=1, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.ReLU(),
                                nn.Conv2d(int(self.K*0.25), self.K, kernel_size=1, stride=1, padding=0, bias=False)]
        else:
            if self.act_type == 'prelu':
                conv_block += [ nn.Conv2d(self.K, self.K, kernel_size=3, stride=self.stride, padding=1, bias=False), 
                                nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.PReLU(self.K),
                                nn.Conv2d(self.K, self.K, kernel_size=1, stride=1, padding=0, bias=False)]
            else:
                conv_block += [ nn.Conv2d(self.K, self.K, kernel_size=3, stride=self.stride, padding=1, bias=False), 
                                nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.ReLU(),
                                nn.Conv2d(self.K, self.K, kernel_size=1, stride=1, padding=0, bias=False)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        act1 = self.act_stem(self.bn_stem(x))
        if self.red:
            out = self.conv_red(act1) + self.conv_block(x)  # add skip connections
        else:
            out = x + self.conv_block(act1)  # add skip connections
        return out

class ResnetBlock_V3(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_c, K, stride, act_type='prelu', red=False, bnEps=2e-5, bnMom=0.9, bottle_neck=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_V3, self).__init__()
        self.in_c = in_c
        self.K = K
        self.stride = stride
        self.act_type = act_type
        self.red = red
        self.bnEps = bnEps
        self.bnMom = bnMom
        self.bottle_neck = bottle_neck
        self.bn_stem = nn.BatchNorm2d(self.in_c, eps=self.bnEps, momentum=self.bnMom, affine=True)
        if self.red:
            self.conv_red = nn.Conv2d(in_c, K, kernel_size=1, stride=self.stride, padding=0, bias=False)
            self.bn_red = nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True)
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        conv_block = []

        if self.bottle_neck:
            if self.act_type == 'prelu':
                conv_block += [ nn.Conv2d(self.in_c, int(self.K*0.25), kernel_size=1, padding=0, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.PReLU(int(self.K*0.25))]

                conv_block += [ nn.Conv2d(int(self.K*0.25), int(self.K*0.25), kernel_size=3, padding=1, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.PReLU(int(self.K*0.25)),
                                nn.Conv2d(int(self.K*0.25), self.K, kernel_size=1, stride=self.stride, padding=0, bias=False)]
            else:
                conv_block += [ nn.Conv2d(self.in_c, int(self.K*0.25), kernel_size=1, padding=0, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.ReLU()]

                conv_block += [ nn.Conv2d(int(self.K*0.25), int(self.K*0.25), kernel_size=3, padding=1, bias=False), 
                                nn.BatchNorm2d(int(self.K*0.25), eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.ReLU(),
                                nn.Conv2d(int(self.K*0.25), self.K, kernel_size=1, stride=self.stride, padding=0, bias=False)]
        else:
            if self.act_type == 'prelu':
                conv_block += [ nn.Conv2d(self.in_c, self.K, kernel_size=3, padding=1, bias=False), 
                                nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.PReLU(self.K),
                                nn.Conv2d(self.K, self.K, kernel_size=3, stride=self.stride, padding=1, bias=False),
                                nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True)]
            else:
                conv_block += [ nn.Conv2d(self.in_c, self.K, kernel_size=3, padding=1, bias=False), 
                                nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True), 
                                nn.ReLU(),
                                nn.Conv2d(self.K, self.K, kernel_size=3, stride=self.stride, padding=1, bias=False),
                                nn.BatchNorm2d(self.K, eps=self.bnEps, momentum=self.bnMom, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        bn1 = self.bn_stem(x)
        if self.red:
            out = self.bn_red(self.conv_red(x)) + self.conv_block(bn1)  # add skip connections
        else:
            out = x + self.conv_block(bn1)  # add skip connections
        return out

class PthResNet(nn.Module):
    # this class gives interface for landmarkloss
    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    # in_c: input image channel number, in_s: input image size (square input)
    def __init__(self, in_c, in_s, emb_size, stages, filters, res_ver='v3', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        assert len(stages)+1==len(filters)
        super(PthResNet, self).__init__()
        self.in_c = in_c
        self.in_s = in_s
        self.emb_size = emb_size
        self.stages = stages
        self.filters = filters
        self.bnEps = bnEps
        self.bnMom = bnMom
        self.bottle_neck = bottle_neck
        if res_ver=='v2':
            residual_module = ResnetBlock_V2
        elif res_ver=='v3':
            residual_module = ResnetBlock_V3
        stride = 2
        final_out_size = int(in_s/(2**len(self.stages)))
        # data input
        #data = mx.sym.Variable("data")

        # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
        #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
        self.model = []
        if in_ver=='v1':
            self.model += [nn.Conv2d(self.in_c, self.filters[0], kernel_size=7, stride=2, padding=3, bias=False)]
        elif in_ver=='v2':
            self.model += [nn.Conv2d(self.in_c, self.filters[0], kernel_size=3, stride=1, padding=1, bias=False)]
        self.model += [ nn.BatchNorm2d(self.filters[0], eps=self.bnEps, momentum=self.bnMom, affine=True),
                        nn.PReLU(num_parameters=self.filters[0]) ]        

        # loop over the number of stages
        
        for i in range(0, len(stages)):
            self.model += [residual_module(self.filters[i], self.filters[i+1], stride, act_type='prelu', red=True, bnEps=self.bnEps, bnMom=self.bnMom, bottle_neck=self.bottle_neck)]
            # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                self.model += [residual_module(self.filters[i+1], self.filters[i+1], 1, act_type='prelu', red=False, bnEps=self.bnEps, bnMom=self.bnMom, bottle_neck=self.bottle_neck)]
                # apply a ResNet module

        self.model += [ nn.BatchNorm2d(self.filters[-1], eps=self.bnEps, momentum=self.bnMom, affine=True),
                        nn.PReLU(self.filters[-1])]
        self.model1 = [ nn.Conv2d(self.filters[-1], self.filters[-1], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(self.filters[-1], eps=self.bnEps, momentum=self.bnMom, affine=True),
                        nn.ReLU(),
                        Flatten(),
                        nn.Linear(self.filters[-1]*final_out_size*final_out_size, self.emb_size, bias=True),
                        nn.BatchNorm1d(self.emb_size, eps=self.bnEps, momentum=self.bnMom, affine=True)]
        self.model = nn.Sequential(*self.model)
        self.model1 = nn.Sequential(*self.model1)

    def forward(self, x):
        """Forward function (with skip connections)"""
        act2_1 = self.model(x)
        embedding = self.model1(act2_1)
        return embedding, act2_1


class PthResNetSimple(nn.Module):
    # this class gives no interface for landmarkloss
    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    # in_c: input image channel number, in_s: input image size (square input)
    def __init__(self, in_c, in_s, emb_size, stages, filters, res_ver='v3', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        assert len(stages)+1==len(filters)
        super(PthResNetSimple, self).__init__()
        self.in_c = in_c
        self.in_s = in_s
        self.emb_size = emb_size
        self.stages = stages
        self.filters = filters
        self.bnEps = bnEps
        self.bnMom = bnMom
        self.bottle_neck = bottle_neck
        if res_ver=='v2':
            residual_module = ResnetBlock_V2
        elif res_ver=='v3':
            residual_module = ResnetBlock_V3
        stride = 2
        final_out_size = int(in_s/(2**len(self.stages)))
        # data input
        #data = mx.sym.Variable("data")

        # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
        #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
        self.model = []
        if in_ver=='v1':
            self.model += [nn.Conv2d(self.in_c, self.filters[0], kernel_size=7, stride=2, padding=3, bias=False)]
        elif in_ver=='v2':
            self.model += [nn.Conv2d(self.in_c, self.filters[0], kernel_size=3, stride=1, padding=1, bias=False)]
        self.model += [ nn.BatchNorm2d(self.filters[0], eps=self.bnEps, momentum=self.bnMom, affine=True),
                        nn.PReLU(num_parameters=self.filters[0]) ]        

        # loop over the number of stages
        
        for i in range(0, len(stages)):
            self.model += [residual_module(self.filters[i], self.filters[i+1], stride, act_type='prelu', red=True, bnEps=self.bnEps, bnMom=self.bnMom, bottle_neck=self.bottle_neck)]
            # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                self.model += [residual_module(self.filters[i+1], self.filters[i+1], 1, act_type='prelu', red=False, bnEps=self.bnEps, bnMom=self.bnMom, bottle_neck=self.bottle_neck)]
                # apply a ResNet module

        self.model += [ nn.BatchNorm2d(self.filters[-1], eps=self.bnEps, momentum=self.bnMom, affine=True),
                        nn.PReLU(self.filters[-1]),
                        nn.Conv2d(self.filters[-1], self.filters[-1], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(self.filters[-1], eps=self.bnEps, momentum=self.bnMom, affine=True),
                        nn.ReLU(),
                        pthlayer.Flatten(),
                        nn.Linear(self.filters[-1]*final_out_size*final_out_size, self.emb_size, bias=True),
                        nn.BatchNorm1d(self.emb_size, eps=self.bnEps, momentum=self.bnMom, affine=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """Forward function (with skip connections)"""
        embedding = self.model(x)
        return embedding