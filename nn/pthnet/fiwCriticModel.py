import torch
from torch import nn, Tensor
import torchvision
import itertools
from .pthlayer import BaseModel
from .pthcritic import CriticModel_FIW
from . import pthutils
from .pthutils import ImagePool
from ..pthloss import pthloss
import fastai
import functools
import time
import numpy as np

class FiwCriticModel(BaseModel):
    """
    This class implements Colorizing GAN during pretrain stage
    """
    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.bn_types = (torch.nn.modules.batchnorm.BatchNorm1d,
                         torch.nn.modules.batchnorm.BatchNorm2d,
                         torch.nn.modules.batchnorm.BatchNorm3d)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['pair', 'unpair', 'total', 'acc_p', 'acc_u']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['pair1', 'pair2', 'unpair']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['Critic']
        self.model_names_for_save = self.model_names

        # define discriminator networks
        self.netCritic = CriticModel_FIW(opt.ModelCritic).to(self.device)
        self.netCritic_layer_groups = self.split(self.netCritic, opt.ModelCritic.split_on)
        #if self.isTrain:  # define discriminators
        #    self.netFt = FeatureExtractorModel(opt.ModelFtExtractor).to(self.device)

        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.pairLoss = pthloss.AdaptiveLoss(nn.BCEWithLogitsLoss())
            self.unpairLoss = pthloss.AdaptiveLoss(nn.BCEWithLogitsLoss())
            self.optimizer = self.create_opt(self.opt.ModelCritic, self.netCritic_layer_groups)
        else:
            self.sigmoid = nn.Sigmoid()

        if self.isTrain:
            self.netCritic.train()
        else:
            self.netCritic.eval()

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #if self.isTrain:
        #    self.schedulers = [pthutils.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        #if self.freeze_gan_encoder:
        #    self.netGan.freeze_encoder()

        self.print_networks(opt.verbose)

    # https://github.com/fastai/fastai/blob/master/fastai/basic_train.py
    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer group `n`."
        for g in self.netCritic_layer_groups[:n]:
            for l in g:
                if not self.opt.train_bn or not isinstance(l, self.bn_types): fastai.torch_core.requires_grad(l, False)
        for g in self.netCritic_layer_groups[n:]: fastai.torch_core.requires_grad(g, True)

    # https://github.com/fastai/fastai/blob/master/fastai/basic_train.py
    def freeze(self)->None:
        "Freeze first layer group."
        assert(len(self.netCritic_layer_groups)>1)
        self.freeze_to(-1)

    # https://github.com/fastai/fastai/blob/master/fastai/basic_train.py
    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def set_input(self, input):
        self.pair1  = input['P1'].to(self.device)
        self.pair2  = input['P2'].to(self.device)
        self.unpair = input['U'].to(self.device)
        self.path_pair1  = input['P1_paths']
        self.path_pair2  = input['P2_paths']
        self.path_unpair = input['U_paths']

    def forward(self, shuffle=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_pair, self.pred_unpair = self.netCritic.forward([self.pair1, self.pair2, self.unpair], shuffle=shuffle)

        if not self.isTrain:
            self.pred_pair = self.sigmoid(self.pred_pair)
            self.pred_unpair = self.sigmoid(self.pred_unpair)
        #print('pred_pair1', self.pred_pair1)
        #print('pred_unpair', self.pred_unpair)

    def backward(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_unpair = self.unpairLoss(self.pred_unpair, self.pred_unpair.new_zeros(self.pred_unpair.shape[0]))
        self.loss_pair = self.pairLoss(self.pred_pair, self.pred_pair.new_ones(self.pred_pair.shape[0]))

        self.loss_total = self.loss_pair*self.opt.LossCoef.pair + self.loss_unpair*self.opt.LossCoef.unpair
        self.loss_total.backward()

        self.loss_acc_u = (self.pred_unpair<0.5).float().sum()/(1.0*self.pred_unpair.shape[0])
        self.loss_acc_p = (self.pred_pair>0.5).float().sum()/(1.0*self.pred_pair.shape[0])

    def forward_backup(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_pair, self.pred_unpair1, self.pred_unpair2 = self.netCritic.forward([self.pair1, self.pair2, self.unpair])
        #print('pred_pair1', self.pred_pair1)
        #print('pred_unpair', self.pred_unpair)

    def backward_backup(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_unpair = (self.unpairLoss(self.pred_unpair1, self.pred_unpair1.new_zeros(self.pred_unpair1.shape[0])) + \
                         self.unpairLoss(self.pred_unpair2, self.pred_unpair2.new_zeros(self.pred_unpair2.shape[0]))) * 0.5

        self.loss_pair = self.pairLoss(self.pred_pair, self.pred_pair.new_ones(self.pred_pair.shape[0]))

        self.loss_total = self.loss_pair*self.opt.LossCoef.pair + self.loss_unpair*self.opt.LossCoef.unpair
        self.loss_total.backward()

        self.loss_acc_u = ((self.pred_unpair1<0.5).float().sum()+(self.pred_unpair2<0.5).float().sum())/(2.0*self.pred_unpair1.shape[0])
        self.loss_acc_p = (self.pred_pair>0.5).float().sum()/(1.0*self.pred_pair.shape[0])

        '''
        if self.opt.discrim_input_size == self.real_A.shape[-1]:
            self.loss_feature = self.featureLoss(self.fake_B, self.real_B)
        else:
            fake_B = torch.nn.functional.interpolate(self.fake_B, size=self.opt.discrim_input_size, mode='bilinear')
            real_B = torch.nn.functional.interpolate(self.real_B, size=self.opt.discrim_input_size, mode='bilinear')
            self.loss_feature = self.featureLoss(fake_B, real_B)
        self.loss_feature = torch.autograd.Variable(self.loss_feature, requires_grad = True) # if not do this, it will report error "RuntimeError: element 0 of variables does not require grad and does not have a grad_fn" when config.freeze_gan_encoder = True
        self.loss_feature.backward()
        '''
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # backward and optimize
        #self.set_requires_grad([self.netFt], False)  # Ds require no gradients when optimizing Gs
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()             # calculate gradients for G_A and G_B
        self.optimizer.step()       # update G_A and G_B's weights

    # use one_cycle_scheduler, original learning_rate will be override
    def train(self, lr_max, dl, freeze_encoder=True, visualizer=None):
        total_iters = 0                # the total number of training iterations
        dataset_size = len(dl)
        if freeze_encoder:
            self.freeze_to(-1) # don't train decoder portion
        else:
            self.unfreeze()
        one_cycle_scheduler = pthutils.OneCycleScheduler(self.optimizer, dl, lr_max)
        one_cycle_scheduler.on_train_begin(n_epochs=self.opt.niter + self.opt.niter_decay + 1, epoch=self.opt.epoch_count)
        for epoch in range(one_cycle_scheduler.start_epoch, one_cycle_scheduler.tot_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            self.netCritic.eval()
            for i, data in enumerate(dl):     
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                if visualizer is not None:
                    visualizer.reset()
                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                self.set_input(data)
                self.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                print(self.pred_pair)

                if total_iters % self.opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opt.update_html_freq == 0
                    self.compute_visuals()
                    if visualizer is not None:
                        visualizer.display_current_results(self.get_current_visuals(), epoch, save_result)

                if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = self.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                    if visualizer is not None:
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                        if self.opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                #if total_iters % self.opt.save_latest_freq == 0:   # cache our latest self every <save_latest_freq> iterations
                #    print('saving the latest self (epoch %d, total_iters %d)' % (epoch, total_iters))
                #    save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                #    self.save_networks(save_suffix)

                iter_data_time = time.time()
                one_cycle_scheduler.on_batch_end(train=True)
                
                #print(self.netGan_layer_groups[0][6].state_dict()['weight'][0,0,0,0])
                #print(self.netGan_layer_groups[1][1].state_dict()['weight'][0,0,0,0])
                #print(self.netGan_layer_groups[2][0].state_dict()['weight_orig'][0,0,0,0])
                #print(one_cycle_scheduler.opt.lr)
                
            if epoch % self.opt.save_epoch_freq == 0:              # cache our self every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.save_networks('latest')
                self.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.niter + self.opt.niter_decay, time.time() - epoch_start_time))


    def test(self, dl):
        pred = []
        total_iters = 0   
        dataset_size = len(dl)
        self.netCritic.eval()

        iter_data_time = time.time() 
        for i, data in enumerate(dl):     
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % self.opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += self.opt.batch_size

            self.set_input(data)
            with torch.no_grad():
                self.forward(shuffle=False)   # calculate loss functions, get gradients, update network weights

            print(data['P1_paths'])
            print(self.pred_pair.squeeze().cpu())

            pred += list(np.array(self.pred_pair.detach().squeeze().cpu()))

            if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                print('iters: ', total_iters, ', time: ', t_comp, ', data: ', t_data)

            iter_data_time = time.time()

        return pred
            