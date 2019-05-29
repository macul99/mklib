import mxnet as mx

class MxMobileFaceNet:
    @staticmethod
    def Act(data, act_type, name):
        if act_type in ['prelu','elu','selu','leaky','rrelu']:
            body = mx.sym.LeakyReLU(data = data, act_type=act_type, name=name)
        else:
            body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
        return body

    @staticmethod
    def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, bnEps=2e-5, bnMom=0.9, act_type="prelu", name="", suffix=""):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=bnEps, momentum=bnMom, name='%s%s_batchnorm' %(name, suffix))
        act = MxMobileFaceNet.Act(data=bn, act_type=act_type, name='%s%s_relu' %(name, suffix))
        return act

    @staticmethod
    def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, bnEps=2e-5, bnMom=0.9, name="", suffix=""):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=bnEps, momentum=bnMom, name='%s%s_batchnorm' %(name, suffix))
        return bn

    @staticmethod
    def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name="", suffix=""):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
        return conv

    @staticmethod
    def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, bnEps=2e-5, bnMom=0.9, name="", suffix=""):
        conv = MxMobileFaceNet.Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name='%s%s_conv_sep' %(name, suffix))
        conv_dw = MxMobileFaceNet.Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, bnEps=bnEps, bnMom=bnMom, name='%s%s_conv_dw' %(name, suffix))
        proj = MxMobileFaceNet.Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name='%s%s_conv_proj' %(name, suffix))
        return proj

    @staticmethod
    def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, bnEps=2e-5, bnMom=0.9, name="", suffix=""):
        identity=data
        for i in range(num_block):
            if stride==(1,1):
                shortcut = identity
            else:
                shortcut = MxMobileFaceNet.ConvOnly(identity, num_filter=num_out, kernel=(1, 1), stride=stride, pad=(0, 0), name='%s%s_shortcut' %(name, suffix), suffix='%d'%i)
            conv = MxMobileFaceNet.DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, bnEps=bnEps, bnMom=bnMom, name='%s%s_block' %(name, suffix), suffix='%d'%i)
            identity = conv+shortcut
        return identity

    @staticmethod
    def build_v1(data, embedding_size, bnEps=2e-5, bnMom=0.9):
        # data input
        #data = mx.sym.Variable("data")

        conv_1 = MxMobileFaceNet.Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), bnEps=bnEps, bnMom=bnMom, name="conv_1")
        conv_2_dw = MxMobileFaceNet.Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bnEps=bnEps, bnMom=bnMom, name="conv_2_dw")
        conv_23 = MxMobileFaceNet.DResidual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, bnEps=bnEps, bnMom=bnMom, name="dconv_23")
        conv_3 = MxMobileFaceNet.Residual(conv_23, num_block=4, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, bnEps=bnEps, bnMom=bnMom, name="res_3")
        conv_34 = MxMobileFaceNet.DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, bnEps=bnEps, bnMom=bnMom, name="dconv_34")
        conv_4 = MxMobileFaceNet.Residual(conv_34, num_block=6, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, bnEps=bnEps, bnMom=bnMom, name="res_4")
        conv_45 = MxMobileFaceNet.DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, bnEps=bnEps, bnMom=bnMom, name="dconv_45")
        conv_5 = MxMobileFaceNet.Residual(conv_45, num_block=2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, bnEps=bnEps, bnMom=bnMom, name="res_5")
        conv_6_sep = MxMobileFaceNet.Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name="conv_6sep")

        conv_7 = MxMobileFaceNet.Linear(conv_6_sep, num_filter=512, kernel=(7, 7), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name="gd_conv")

        conv_8 = MxMobileFaceNet.Linear(conv_7, num_filter=embedding_size, kernel=(1, 1), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name="out_conv")

        fc1 = mx.sym.Flatten(data=conv_8, name='out_fc1')
        embedding = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=bnEps, momentum=bnMom, name='out_embedding')

        # return the network architecture
        return embedding, conv_7

    @staticmethod
    def build_v2(data, embedding_size, bnEps=2e-5, bnMom=0.9):
        # data input
        #data = mx.sym.Variable("data")

        conv_1 = MxMobileFaceNet.Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), bnEps=bnEps, bnMom=bnMom, name="conv_1")
        conv_2_dw = MxMobileFaceNet.Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bnEps=bnEps, bnMom=bnMom, name="conv_2_dw")

        conv_3 = MxMobileFaceNet.Residual(conv_2_dw, num_block=1, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, bnEps=bnEps, bnMom=bnMom, name="res_3")

        conv_4 = MxMobileFaceNet.Residual(conv_3, num_block=4, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, bnEps=bnEps, bnMom=bnMom, name="res_4")
        
        conv_5 = MxMobileFaceNet.Residual(conv_4, num_block=1, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, bnEps=bnEps, bnMom=bnMom, name="res_5")

        conv_6 = MxMobileFaceNet.Residual(conv_5, num_block=6, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, bnEps=bnEps, bnMom=bnMom, name="res_6")

        conv_7 = MxMobileFaceNet.Residual(conv_6, num_block=1, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, bnEps=bnEps, bnMom=bnMom, name="res_7")

        conv_8 = MxMobileFaceNet.Residual(conv_7, num_block=2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, bnEps=bnEps, bnMom=bnMom, name="res_8")

        conv_9_sep = MxMobileFaceNet.Conv(conv_8, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name="conv_9sep")

        conv_10 = MxMobileFaceNet.Linear(conv_9_sep, num_filter=512, kernel=(7, 7), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name="gd_conv")

        conv_11 = MxMobileFaceNet.Linear(conv_10, num_filter=embedding_size, kernel=(1, 1), pad=(0, 0), stride=(1, 1), bnEps=bnEps, bnMom=bnMom, name="out_conv")

        fc1 = mx.sym.Flatten(data=conv_11, name='out_fc1')
        embedding = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=bnEps, momentum=bnMom, name='out_embedding')
        
        # return the network architecture
        return embedding, conv_10

