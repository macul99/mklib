import mxnet as mx
import math

class MxLosses:
    @staticmethod
    def Act(data, act_type, name):
        if act_type in ['prelu','elu','selu','leaky','rrelu']:
            body = mx.sym.LeakyReLU(data = data, act_type=act_type, name=name)
        else:
            body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
        return body

    @staticmethod
    def arc_loss(data, label, num_classes, emb_size, mrg_angle, mrg_scale, grad_scale=1.0, act_type='relu', easy_margin=False, name="arc"):
        # check the input range
        assert mrg_scale>0.0
        assert mrg_angle>=0.0
        assert mrg_angle<(math.pi/2)

        label = mx.sym.reshape(label,shape=(-1,)) # change [[0]] to [0]

        _weight = mx.symbol.Variable(name+"_fc_weight", shape=(num_classes, emb_size))
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(data, mode='instance', name='fc1n')*mrg_scale
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=num_classes, name=name+"_fc")
        
        zy = mx.sym.pick(fc7, label, axis=1)
        
        
        cos_t = zy/mrg_scale
        cos_m = math.cos(mrg_angle)
        sin_m = math.sin(mrg_angle)
        mm = math.sin(math.pi-mrg_angle)*mrg_angle
        #threshold = 0.0
        threshold = math.cos(math.pi-mrg_angle)
        if easy_margin:
          cond = MxLosses.Act(cos_t, act_type, name+"_"+act_type)
        else:
          cond_v = cos_t - threshold
          cond = MxLosses.Act(cond_v, act_type, name+"_"+act_type)
        body = cos_t*cos_t
        body = 1.0-body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t*cos_m
        b = sin_t*sin_m
        new_zy = new_zy - b
        new_zy = new_zy*mrg_scale
        if easy_margin:
          zy_keep = zy
        else:
          zy_keep = zy - mrg_scale*mm
        new_zy = mx.sym.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(label, depth = num_classes, on_value = 1.0, off_value = 0.0)
        #gt_one_hot = mx.sym.flatten(gt_one_hot)
        
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        
        fc7 = fc7+body
                
        softmax_out = mx.symbol.SoftmaxOutput(data=fc7, label=label, grad_scale=grad_scale, name='softmax', normalization='valid')
        return softmax_out

    @staticmethod
    def arc_loss_backup(data, label, num_classes, emb_size, mrg_angle, mrg_scale, grad_scale=1.0, act_type='relu', easy_margin=False, name="arc"):
        # check the input range
        assert mrg_scale>0.0
        assert mrg_angle>=0.0
        assert mrg_angle<(math.pi/2)

        _weight = mx.symbol.Variable(name+"_fc_weight", shape=(num_classes, emb_size))
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(data, mode='instance', name='fc1n')*mrg_scale
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=num_classes, name=name+"_fc")
        zy = mx.sym.pick(fc7, label, axis=1)
        cos_t = zy/mrg_scale
        cos_m = math.cos(mrg_angle)
        sin_m = math.sin(mrg_angle)
        mm = math.sin(math.pi-mrg_angle)*mrg_angle
        #threshold = 0.0
        threshold = math.cos(math.pi-mrg_angle)
        if easy_margin:
          cond = MxLosses.Act(cos_t, act_type, name+"_"+act_type)
        else:
          cond_v = cos_t - threshold
          cond = MxLosses.Act(cond_v, act_type, name+"_"+act_type)
        body = cos_t*cos_t
        body = 1.0-body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t*cos_m
        b = sin_t*sin_m
        new_zy = new_zy - b
        new_zy = new_zy*mrg_scale
        if easy_margin:
          zy_keep = zy
        else:
          zy_keep = zy - mrg_scale*mm
        new_zy = mx.sym.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(label, depth = num_classes, on_value = 1.0, off_value = 0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7+body

        softmax_out = mx.symbol.SoftmaxOutput(data=fc7, label=label, grad_scale=grad_scale, name='softmax', normalization='valid')
        return softmax_out


    @staticmethod
    def regression_loss(data, label, num_classes, grad_scale=1.0, act_type='tanh', name="regr"):

        pool1 = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_pool')
        dp1 = mx.symbol.Dropout(data=pool1, p=0.5, name=name+"_dropout")
        fc = mx.sym.FullyConnected(data=dp1, num_hidden=num_classes, name=name+"_fc")        
        fc = MxLosses.Act(fc, act_type, name+"_"+act_type)
        lr_out = mx.symbol.LinearRegressionOutput(data=fc, label=label, grad_scale=grad_scale, name="lro")

        return lr_out

 
    @staticmethod
    def arc_landmark_loss(data, data1, softmax_label, landmark_gt, mrg_angle, mrg_scale, num_classes, emb_size, landmark_len=10, grad_scales=[1.0, 1.0]):
        # label_names should match those specified in ImageIter, label_names[0] is for people classification, label_name[1] is for landmark-5p regression
        # landmark_len = num_points*2
        #assert len(label_names)==2
        #softmax_label = mx.symbol.Variable(label_names[0],init=mx.initializer.Zero())
        #landmark_gt  = mx.symbol.Variable(label_names[1],init=mx.initializer.Zero())
        
        softmax_out = MxLosses.arc_loss(data, softmax_label, num_classes, emb_size, mrg_angle, mrg_scale, grad_scale=grad_scales[0])
        lr_out = MxLosses.regression_loss(data1, landmark_gt, landmark_len, act_type='tanh', grad_scale=grad_scales[1])

        out_list = [softmax_out, lr_out, mx.symbol.BlockGrad(data, name='embedding')] # corresponds to the pred sequence, softmax_out <-> pred[0], lr_out <-> pred[1], data <-> pred[2]

        out = mx.symbol.Group(out_list)
        return out


    @staticmethod
    def arc_loss_only(data, softmax_label, mrg_angle, mrg_scale, num_classes, emb_size, grad_scales=[1.0]):
        # label_names should match those specified in ImageIter, label_names[0] is for people classification, label_name[1] is for landmark-5p regression
        # landmark_len = num_points*2
        #assert len(label_names)==1
        #softmax_label = mx.symbol.Variable(label_names[0])
        
        softmax_out = MxLosses.arc_loss(data, softmax_label, num_classes, emb_size, mrg_angle, mrg_scale, grad_scale=grad_scales[0])

        #out_list = [softmax_out, mx.symbol.BlockGrad(data, name='embedding')] # corresponds to the pred sequence, softmax_out <-> pred[0], lr_out <-> pred[1], data <-> pred[2]

        #out = mx.symbol.Group(out_list)
        return softmax_out
