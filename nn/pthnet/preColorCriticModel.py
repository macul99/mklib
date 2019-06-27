import torch
from torch import nn, Tensor
import torchvision
import itertools
from .pthlayer import BaseModel
from .pthcritic import CriticModel
from . import pthutils
from .pthutils import ImagePool
from ..pthloss import pthloss
import fastai
import functools
import time
import numpy as np

class PreColorCriticModel(BaseModel):
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
        if self.opt.ModelCritic.discriminator_type == 'both':
            self.loss_names = ['gan_pair', 'gan_unpair']
        elif self.opt.ModelCritic.discriminator_type == 'pair':
            self.loss_names = ['gan_pair']
        elif self.opt.ModelCritic.discriminator_type == 'unpair':
            self.loss_names = ['gan_unpair']
        else:
            raise NotImplementedError("")
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []
        if self.opt.ModelCritic.discriminator_type in ['pair', 'both']:
            self.visual_names += ['real_A_pair', 'real_B_pair', 'fake_B_pair']
        if self.opt.ModelCritic.discriminator_type in ['unpair', 'both']:
            self.visual_names += ['real_B_unpair', 'fake_B_unpair']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['Critic']
            self.model_names_for_save = self.model_names
        else:  # during test time, only load Gs
            self.model_names = ['Critic']

        # define discriminator networks
        self.netCritic = CriticModel(opt.ModelCritic).to(self.device)
        self.netCritic_layer_groups = [self.netCritic.encoder]
        self.netCritic_lr = [self.opt.ModelCritic.lr/10.]
        if self.netCritic.discriminator_pair is not None:
            self.netCritic_layer_groups.append(self.netCritic.discriminator_pair)
            self.netCritic_lr.append(self.opt.ModelCritic.lr)
        if self.netCritic.discriminator_unpair is not None:
            self.netCritic_layer_groups.append(self.netCritic.discriminator_unpair)
            self.netCritic_lr.append(self.opt.ModelCritic.lr)
        #if self.isTrain:  # define discriminators
        #    self.netFt = FeatureExtractorModel(opt.ModelFtExtractor).to(self.device)

        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.opt.ModelCritic.discriminator_type in ['pair', 'both']:
                self.ganPairLoss = pthloss.AdaptiveLoss(nn.BCEWithLogitsLoss())
            if self.opt.ModelCritic.discriminator_type in ['unpair', 'both']:
                self.ganUnpairLoss = pthloss.AdaptiveLoss(nn.BCEWithLogitsLoss())
            self.opt.ModelCritic.lr = self.netCritic_lr
            self.optimizer_D = self.create_opt(self.opt.ModelCritic, self.netCritic_layer_groups)
            self.optimizers.append(self.optimizer_D)

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
        self.freeze_to(1)

    # https://github.com/fastai/fastai/blob/master/fastai/basic_train.py
    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def set_input(self):
        pass

    def set_input_pair(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A_pair = input['A'].to(self.device)
        self.real_B_pair = input['B'].to(self.device)
        self.fake_B_pair = input['C'].to(self.device)
        self.path_rA_pair = input['A_paths']
        self.path_rB_pair = input['B_paths']
        self.path_fB_pair = input['C_paths']

    def set_input_unpair(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.fake_B_unpair = input['C'].to(self.device)
        self.real_B_unpair = input['B'].to(self.device)
        self.path_fB_unpair = input['C_paths']
        self.path_rB_unpair = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.ModelCritic.discriminator_type in ['pair', 'both']:
            self.pred_fake_pair = self.netCritic.forward_pair([self.real_A_pair, self.fake_B_pair])
            self.pred_real_pair = self.netCritic.forward_pair([self.real_A_pair, self.real_B_pair])

        if self.opt.ModelCritic.discriminator_type in ['unpair', 'both']:
            self.pred_fake_unpair = self.netCritic.forward_unpair(self.fake_B_unpair)
            self.pred_real_unpair = self.netCritic.forward_unpair(self.real_B_unpair)

    def backward_D(self):
        """Calculate the loss for generators G_A and G_B"""
        if self.opt.ModelCritic.discriminator_type in ['pair', 'both']:
            self.loss_gan_pair = self.ganPairLoss(self.pred_fake_pair, self.pred_fake_pair.new_zeros(self.pred_fake_pair.shape[0])) + \
                                 self.ganPairLoss(self.pred_real_pair, self.pred_real_pair.new_ones(self.pred_fake_pair.shape[0]))

        if self.opt.ModelCritic.discriminator_type in ['unpair', 'both']:
            self.loss_gan_unpair = self.ganUnpairLoss(self.pred_fake_unpair, self.pred_fake_unpair.new_zeros(self.pred_fake_unpair.shape[0])) + \
                                   self.ganUnpairLoss(self.pred_real_unpair, self.pred_real_unpair.new_ones(self.pred_fake_unpair.shape[0]))

        if self.opt.ModelCritic.discriminator_type == 'pair':
            self.loss_gan = self.loss_gan_pair*self.opt.LossCoef.gan_pair_D
        elif self.opt.ModelCritic.discriminator_type == 'unpair':
            self.loss_gan = self.loss_gan_unpair*self.opt.LossCoef.gan_unpair_D
        else: # both
            self.loss_gan = self.loss_gan_pair*self.opt.LossCoef.gan_pair_D + self.loss_gan_unpair*self.opt.LossCoef.gan_unpair_D
        self.loss_gan.backward()
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
        self.optimizer_D.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_D()             # calculate gradients for G_A and G_B
        self.optimizer_D.step()       # update G_A and G_B's weights

    # use one_cycle_scheduler, original learning_rate will be override
    def train(self, lr_max, dl_pair, dl_unpair, freeze_encoder=True, visualizer=None):
        assert self.opt.ModelCritic.discriminator_type == 'both'
        assert dl_pair is not None and dl_unpair is not None

        if len(dl_pair) >= len(dl_unpair):
            dl_more = dl_pair
            dl_less = dl_unpair
            dl_flag = 'pair'
        else:
            dl_more = dl_unpair
            dl_less = dl_pair
            dl_flag = 'unpair'

        total_iters = 0                # the total number of training iterations
        dataset_size = len(dl_more)
        if freeze_encoder:
            self.freeze_to(1) # don't train decoder portion
        else:
            self.unfreeze()
        one_cycle_scheduler = pthutils.OneCycleScheduler(self.optimizer_D, dl_more, lr_max)
        one_cycle_scheduler.on_train_begin(n_epochs=self.opt.niter + self.opt.niter_decay + 1, epoch=self.opt.epoch_count)
        for epoch in range(one_cycle_scheduler.start_epoch, one_cycle_scheduler.tot_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            self.netCritic.train()
            for i, data in enumerate(zip(itertools.cycle(dl_less), dl_more)):     
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                if visualizer is not None:
                    visualizer.reset()
                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                if dl_flag == 'pair':
                    self.set_input_unpair(data[0])         # unpack data from dataset and apply preprocessing
                    self.set_input_pair(data[1])         # unpack data from dataset and apply preprocessing
                else:
                    self.set_input_pair(data[0])         # unpack data from dataset and apply preprocessing
                    self.set_input_unpair(data[1])         # unpack data from dataset and apply preprocessing
                self.optimize_parameters()   # calculate loss functions, get gradients, update network weights

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


    # use one_cycle_scheduler, original learning_rate will be override
    def train_single_dl(self, lr_max, dl, freeze_encoder=True, visualizer=None):
        assert self.opt.ModelCritic.discriminator_type == 'pair' or self.opt.ModelCritic.discriminator_type == 'unpair'        
        assert dl is not None

        if self.opt.ModelCritic.discriminator_type == 'pair':
            set_input = self.set_input_pair
        else:
            set_input = self.set_input_unpair

        total_iters = 0                # the total number of training iterations
        dataset_size = len(dl)
        if freeze_encoder:
            self.freeze_to(1) # don't train decoder portion
        else:
            self.unfreeze()
        one_cycle_scheduler = pthutils.OneCycleScheduler(self.optimizer_D, dl, lr_max)
        one_cycle_scheduler.on_train_begin(n_epochs=self.opt.niter + self.opt.niter_decay + 1, epoch=self.opt.epoch_count)
        for epoch in range(one_cycle_scheduler.start_epoch, one_cycle_scheduler.tot_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            self.netCritic.train()
            for i, data in enumerate(dl):     
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                if visualizer is not None:
                    visualizer.reset()
                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                set_input(data)
                self.optimize_parameters()   # calculate loss functions, get gradients, update network weights

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