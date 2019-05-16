# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/deepinsight/insightface/blob/master/src/symbols/fresnet.py

Implemented the following paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''

import mxnet as mx

class MxResNet:
    @staticmethod
    def Act(data, act_type, name):
        if act_type in ['prelu','elu','selu','leaky','rrelu']:
            body = mx.sym.LeakyReLU(data = data, act_type=act_type, name=name)
        else:
            body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
        return body

    @staticmethod
    def residual_module_v2(data, K, stride, act_type='prelu', red=False, bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, name=""):
        # the shortcut branch of the ResNet module should be initialized as the input (identity) data
        shortcut = data
        
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn1')
        act1 = MxResNet.Act(data=bn1, act_type=act_type, name=name + '_%s1' % (act_type))
        if bottle_neck:
            # the first block of the ResNet module are 1x1 CONVs
            conv1 = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=int(K * 0.25), no_bias=True, name=name + '_conv1')
            # the second block of the ResNet module are 3x3 CONVs
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn2')
            act2 = MxResNet.Act(data=bn2, act_type=act_type, name=name + '_%s2' % (act_type))
            conv2 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=stride, num_filter=int(K * 0.25), no_bias=True, name=name + '_conv2')

            # the third block of the ResNet module is another set of 1x1 CONVs
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn3')
            act3 = MxResNet.Act(data=bn3, act_type=act_type, name=name + '_%s3' % (act_type))
            conv3 = mx.sym.Convolution(data=act3, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=K, no_bias=True, name=name + '_conv3')
        else:
            conv1 = mx.sym.Convolution(data=act1, pad=(1, 1), kernel=(3, 3), stride=stride, num_filter=K, no_bias=True, name=name + '_conv1')

            # the second block of the ResNet module are 3x3 CONVs
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn2')
            act2 = MxResNet.Act(data=bn2, act_type=act_type, name=name + '_%s2' % (act_type))
            conv3 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=K, no_bias=True, name=name + '_conv2')


        # add Squeeze-and-Excitation module
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=conv3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
          body = mx.sym.Convolution(data=body, num_filter=K//16, kernel=(1,1), stride=(1,1), pad=(0,0), name=name + '_se_conv1')
          body = MxResNet.Act(data=body, act_type=act_type, name=name + '_se_%s1' % (act_type))
          body = mx.sym.Convolution(data=body, num_filter=K, kernel=(1,1), stride=(1,1), pad=(0,0), name=name + '_se_conv2')
          body = MxResNet.Act(data=body, act_type='sigmoid', name=name + '_se_sigmoid1')
          conv3 = mx.symbol.broadcast_mul(conv3, body, name=name + '_se_bc1')

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = mx.sym.Convolution(data=act1, pad=(0, 0), kernel=(1, 1), stride=stride, num_filter=K, no_bias=True, name=name + '_convr')

        # add together the shortcut and the final CONV
        add = conv3 + shortcut

        # return the addition as the output of the ResNet module
        return add

    @staticmethod
    def residual_module_v3(data, K, stride, act_type='prelu', red=False, bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False, name=""):
        # the shortcut branch of the ResNet module should be initialized as the input (identity) data
        shortcut = data

        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn1')
        if bottle_neck:
            # the first block of the ResNet module are 1x1 CONVs
            conv1 = mx.sym.Convolution(data=bn1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=int(K * 0.25), no_bias=True, name=name + '_conv1')

            # the second block of the ResNet module are 3x3 CONVs
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn2')
            act2 = MxResNet.Act(data=bn2, act_type=act_type, name=name + '_%s2' % (act_type))
            conv2 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=int(K * 0.25), no_bias=True, name=name + '_conv2')

            # the third block of the ResNet module is another set of 1x1 CONVs
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn3')
            act3 = MxResNet.Act(data=bn3, act_type=act_type, name=name + '_%s3' % (act_type))
            conv3 = mx.sym.Convolution(data=act3, pad=(0, 0), kernel=(1, 1), stride=stride, num_filter=K, no_bias=True, name=name + '_conv3')

            # the forth block for bn
            bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn4')
        else:
            conv1 = mx.sym.Convolution(data=bn1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=K, no_bias=True, name=name + '_conv1')

            # the second block of the ResNet module are 3x3 CONVs
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn2')
            act2 = MxResNet.Act(data=bn2, act_type=act_type, name=name + '_%s2' % (act_type))
            conv2 = mx.sym.Convolution(data=act2, pad=(1, 1), kernel=(3, 3), stride=stride, num_filter=K, no_bias=True, name=name + '_conv2')

            # the third block of the ResNet module is another set of 1x1 CONVs
            bn4 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bn4')

        # add Squeeze-and-Excitation module
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
          body = mx.sym.Convolution(data=body, num_filter=K//16, kernel=(1,1), stride=(1,1), pad=(0,0), name=name + '_se_conv1')
          body = MxResNet.Act(data=body, act_type=act_type, name=name + '_se_%s1' % (act_type))
          body = mx.sym.Convolution(data=body, num_filter=K, kernel=(1,1), stride=(1,1), pad=(0,0), name=name + '_se_conv2')
          body = MxResNet.Act(data=body, act_type='sigmoid', name=name + '_se_sigmoid1')
          bn4 = mx.symbol.broadcast_mul(bn4, body, name=name + '_se_bc1')

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = mx.sym.Convolution(data=data, pad=(0, 0), kernel=(1, 1), stride=stride, num_filter=K, no_bias=True, name=name + '_convr')
            shortcut = mx.sym.BatchNorm(data=shortcut, fix_gamma=False, eps=bnEps, momentum=bnMom, name=name + '_bnr')

        # add together the shortcut and the final CONV
        add = bn4 + shortcut

        # return the addition as the output of the ResNet module
        return add

    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    @staticmethod
    def build(data, classes, stages, filters, res_ver='v2', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        if res_ver=='v2':
            residual_module = MxResNet.residual_module_v2
        elif res_ver=='v3':
            residual_module = MxResNet.residual_module_v3
        # data input
        #data = mx.sym.Variable("data")

        # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
        #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
        if in_ver=='v1':
            conv1_1 = mx.sym.Convolution(data=data, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=filters[0], no_bias=True, name='stem_conv1')
        elif in_ver=='v2':
            conv1_1 = mx.sym.Convolution(data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=filters[0], no_bias=True, name='stem_conv1')
        bn1_2 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=bnEps, momentum=bnMom, name='stem_bn2')
        act1_2 = MxResNet.Act(data=bn1_2, act_type="prelu", name='stem_relu1')
        #pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
        body = act1_2

        # loop over the number of stages
        stride = (2, 2)
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            
            body = residual_module(body, filters[i + 1], stride, red=True, bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage%d_unit%d' % (i+1, 1))

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                body = residual_module(body, filters[i + 1], (1, 1), bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage%d_unit%d' % (i+1, j+2))


        bn2_1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=bnEps, momentum=bnMom, name='out_bn2')
        act2_1 = MxResNet.Act(data=bn2_1, act_type="prelu", name='out_relu2')
        conv2_1 = mx.sym.Convolution(data=act2_1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=filters[-1], no_bias=True, name='out_conv1')
        # apply BN => ACT => POOL
        bn3_1 = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=bnEps, momentum=bnMom, name='out_bn3')
        act3_1 = MxResNet.Act(data=bn3_1, act_type="relu", name='out_relu3')

        # embedding
        #flatten = mx.sym.Flatten(data=act2_1, name='out_ft1') # don't use flatten here, the parameter will be much bigger
        fc1 = mx.sym.FullyConnected(data=act3_1, num_hidden=classes, name='out_fc1')
        embedding = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=bnEps, momentum=bnMom, name='out_embedding')

        # return the network architecture
        return embedding, act2_1

    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    @staticmethod
    def build_for_landmarkloss(data, classes, stages, filters, stage_lm=2, module_num_lm=2, res_ver='v2', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        if res_ver=='v2':
            residual_module = MxResNet.residual_module_v2
        elif res_ver=='v3':
            residual_module = MxResNet.residual_module_v3
        # data input
        #data = mx.sym.Variable("data")

        # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
        #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
        if in_ver=='v1':
            conv1_1 = mx.sym.Convolution(data=data, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=filters[0], no_bias=True, name='stem_conv1')
        elif in_ver=='v2':
            conv1_1 = mx.sym.Convolution(data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=filters[0], no_bias=True, name='stem_conv1')
        bn1_2 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=bnEps, momentum=bnMom, name='stem_bn2')
        act1_2 = MxResNet.Act(data=bn1_2, act_type="prelu", name='stem_relu1')
        #pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
        body = act1_2

        # loop over the number of stages
        stride = (2, 2)
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            
            body = residual_module(body, filters[i + 1], stride, red=True, bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage%d_unit%d' % (i+1, 1))

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                body = residual_module(body, filters[i + 1], (1, 1), bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage%d_unit%d' % (i+1, j+2))

            if i==stage_lm-1:
                body_lm = body

        # build landmark loss branch
        body_lm = residual_module(body_lm, filters[stage_lm + 1], stride, red=True, bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage_lm_unit%d' % (1))

        # loop over the number of layers in the stage
        for j in range(0, module_num_lm):
            # apply a ResNet module
            body_lm = residual_module(body_lm, filters[stage_lm+1], (1, 1), bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage_lm_unit%d' % (j+2))


        body_lm = mx.sym.BatchNorm(data=body_lm, fix_gamma=False, eps=bnEps, momentum=bnMom, name='out_bn2')
        body_lm = MxResNet.Act(data=body_lm, act_type="prelu", name='out_relu2')

        # embedding
        #flatten = mx.sym.Flatten(data=act2_1, name='out_ft1') # don't use flatten here, the parameter will be much bigger
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=bnEps, momentum=bnMom, name='out_bn3')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=classes, name='out_fc1')
        embedding = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=bnEps, momentum=bnMom, name='out_embedding')

        # return the network architecture
        return embedding, body_lm


    # for resnet: stages=(3,4,6,3), filters=(64,256,512,1024,2048)
    @staticmethod
    def build_arcloss_only(data, classes, stages, filters, res_ver='v2', in_ver='v2', bnEps=2e-5, bnMom=0.9, bottle_neck=False, use_se=False):
        assert res_ver in ['v2','v3'], 'only residuel module version 2 and 3 are supported!!'
        assert in_ver in ['v1','v2'], 'only input version 1 and 2 are supported!!'
        if res_ver=='v2':
            residual_module = MxResNet.residual_module_v2
        elif res_ver=='v3':
            residual_module = MxResNet.residual_module_v3
        # data input
        #data = mx.sym.Variable("data")

        # Block #1: BN => CONV => ACT => POOL, then initialize the "body" of the network
        #bn1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bnEps, momentum=bnMom, name='stem_bn1')
        if in_ver=='v1':
            conv1_1 = mx.sym.Convolution(data=data, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=filters[0], no_bias=True, name='stem_conv1')
        elif in_ver=='v2':
            conv1_1 = mx.sym.Convolution(data=data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=filters[0], no_bias=True, name='stem_conv1')
        bn1_2 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=bnEps, momentum=bnMom, name='stem_bn2')
        act1_2 = MxResNet.Act(data=bn1_2, act_type="prelu", name='stem_relu1')
        #pool1 = mx.sym.Pooling(data=act1_2, pool_type="max", pad=(1, 1), kernel=(3, 3), stride=(2, 2))
        body = act1_2

        # loop over the number of stages
        stride = (2, 2)
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume            
            body = residual_module(body, filters[i + 1], stride, red=True, bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage%d_unit%d' % (i+1, 1))

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                body = residual_module(body, filters[i + 1], (1, 1), bnEps=bnEps, bnMom=bnMom, bottle_neck=bottle_neck, use_se=use_se, name='stage%d_unit%d' % (i+1, j+2))

        # embedding
        #flatten = mx.sym.Flatten(data=act2_1, name='out_ft1') # don't use flatten here, the parameter will be much bigger
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=bnEps, momentum=bnMom, name='out_bn3')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=classes, name='out_fc1')
        embedding = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=bnEps, momentum=bnMom, name='out_embedding')

        # return the network architecture
        return embedding
 