import torch
import torchvision
import itertools
from .pthlayer import BaseModel
from .pthunet import UnetWideModel, FeatureExtractorModel
from . import pthutils
from .pthutils import ImagePool
from ..pthloss import pthloss
import fastai
import functools
import time
from os import makedirs
from os.path import join, exists, abspath

class PreColorGenerateModel(BaseModel):
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
        #self.freeze_gan_encoder = opt.freeze_gan_encoder

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['feature']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        else:
            self.visual_names = ['real_A', 'fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['Gan']
            self.model_names_for_save = self.model_names
        else:  # during test time, only load Gs
            self.model_names = ['Gan']

        # define generator networks
        self.netGan = UnetWideModel(opt.ModelGan).to(self.device)
        self.netGan_layer_groups = self.split(self.netGan.model, opt.ModelGan.split_on)
        #if self.isTrain:  # define discriminators
        #    self.netFt = FeatureExtractorModel(opt.ModelFtExtractor).to(self.device)

        if self.isTrain:
            # define loss functions
            if self.opt.feature_loss_name == 'vgg16':
                self.featureLoss = pthloss.FeatureLoss_Vgg16(self.device)
            elif self.opt.feature_loss_name == 'resnet20':
                self.featureLoss = pthloss.FeatureLoss_Resnet20(self.device, model_path=self.opt.resnet20_model_path)
            else:
                raise NotImplementedError("")
            self.pixelLoss = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = self.create_opt(self.opt.ModelGan, self.netGan_layer_groups)
            self.optimizers.append(self.optimizer_G)

        if self.isTrain:
            self.netGan.train()
        else:
            self.netGan.eval()

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
        for g in self.netGan_layer_groups[:n]:
            for l in g:
                if not self.opt.train_bn or not isinstance(l, self.bn_types): fastai.torch_core.requires_grad(l, False)
        for g in self.netGan_layer_groups[n:]: fastai.torch_core.requires_grad(g, True)

    # https://github.com/fastai/fastai/blob/master/fastai/basic_train.py
    def freeze(self)->None:
        "Freeze up to last layer group."
        assert(len(self.netGan_layer_groups)>1)
        self.freeze_to(-1)

    # https://github.com/fastai/fastai/blob/master/fastai/basic_train.py
    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.path_A = input['A_paths']
        self.path_B = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netGan(self.real_A)  # G_A(A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_feature = self.featureLoss(self.fake_B, self.real_B)
        self.loss_feature.backward()

        self.loss_pixel = self.pixelLoss(self.fake_B, self.real_B)
        self.loss_pixel.backward()
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
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

    # use one_cycle_scheduler, original learning_rate will be override
    def train(self, lr_max, dl, freeze_encoder=True, visualizer=None):
        total_iters = 0                # the total number of training iterations
        dataset_size = len(dl)
        if freeze_encoder:
            self.freeze_to(1) # don't train decoder portion
        else:
            self.unfreeze()
        one_cycle_scheduler = pthutils.OneCycleScheduler(self.optimizer_G, dl, lr_max)
        one_cycle_scheduler.on_train_begin(n_epochs=self.opt.niter + self.opt.niter_decay + 1, epoch=self.opt.epoch_count)
        for epoch in range(one_cycle_scheduler.start_epoch, one_cycle_scheduler.tot_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            self.netGan.train()
            for i, data in enumerate(dl):  # inner loop within one epoch        
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                if visualizer is not None:
                    visualizer.reset()
                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                self.set_input(data)         # unpack data from dataset and apply preprocessing
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

    # the final path will be 'save_path/org_folder_level[-2]/org_folder_level[-1]/file_name' if folderlevel=2
    def generate_image(self, dl, save_path):
        if not exists(save_path):
            makedirs(save_path)
        self.netGan.eval()
        for i, data in enumerate(dl):
            self.set_input(data)         # unpack data from dataset and apply preprocessing
            self.forward() # self.fake_B is generated
            for j, (fb, pa) in enumerate(zip(self.fake_B, self.path_A)):
                im = dl.dataset.reconstruct_B(fb)
                tmpPath = [save_path] + pa.split('/')[-1-dl.dataset_opt.folderlevel:]
                tmpFolder = join(*tmpPath[:-1])
                if not exists(tmpFolder):
                    makedirs(tmpFolder)
                dl.dataset.save_as_image(im, join(*tmpPath))
