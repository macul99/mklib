import torch
from torch import Tensor
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
import os
from os import mkdir, makedirs, rename, listdir
from os.path import join, exists, relpath, abspath
import importlib
from . import pthutils

##############################################################################
# derived from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    # denormalize image tensor and convert to np array with value of 0-255
    def reconstruct(self, t:Tensor, denorm_fn): 
        x = denorm_fn(t.detach().clone())
        x = x.float().clamp(min=0,max=1)
        return pthutils.image2np(x*255).astype(np.uint8)

    def save_as_image(self, x, file_name):
        Image.fromarray(x).save(file_name)


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/base_dataset.py
# get_params()
def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/base_dataset.py
# get_transform()
def get_transform(opt, params=None, grayscale=False, method=Image.BILINEAR, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    # if opt.preprocess == 'none':
    #    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((opt.mean[0],), (opt.std[0],))]
        else:
            transform_list += [transforms.Normalize(opt.mean, opt.std)] # imagenet stats
    '''
    if convert:        
        if grayscale:
            transform_list += [lambda x: torch.from_numpy((np.array(x).astype(np.float32)-mean[0,0,0])*0.0078125)]
        else:
            transform_list += [lambda x: torch.from_numpy((np.array(x).astype(np.float32)-mean)*0.0078125)]
    '''
    return transforms.Compose(transform_list)


def get_denorm(mean, std, grayscale=False):
    mean = np.array(mean)
    std = np.array(std)
    if grayscale:
        return transforms.Normalize((-mean[0]/std[0],),(1/std[0],))
    else:
        return transforms.Normalize(-mean/std, 1/std)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
##############################################################################


# For paired dataset
# Group A is NIR images
# Group B is VIS images
# Group C is generated images
# Label is the person ID
# It is called paired because the image pose, expression is similar for the paired images between A and B
class PairedLabelDataset(BaseDataset):
    """paired dataset with label
    """

    def __init__(self, opt, seed=100):
        BaseDataset.__init__(self, opt)
        if self.opt.dataset_name == 'oulu':
            self.__init_oulu()
        else:
            raise NotImplementedError
        random.seed(seed)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, Image.open(self.A_path[0]).convert('RGB').size)
        self.A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_params = get_params(self.opt, Image.open(self.B_path[0]).convert('RGB').size)
        self.B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        self.denorm_A = get_denorm(opt.mean, opt.std, self.input_nc==1) # denorm transformation
        self.denorm_B = get_denorm(opt.mean, opt.std, self.output_nc==1) # denorm transformation

        if self.opt.generatedroot is not None:
            transform_params = get_params(self.opt, Image.open(self.C_path[0]).convert('RGB').size)
            self.C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            self.denorm_C = get_denorm(opt.mean, opt.std, self.output_nc==1) # denorm transformation

    def __init_oulu(self):
        dir_A = join(self.root, 'NI/Strong')
        dir_B = join(self.root, 'VL/Strong')
        ds_A = sorted(make_dataset(dir_A, self.opt.max_dataset_size))
        ds_B = sorted(make_dataset(dir_B, self.opt.max_dataset_size))
        assert len(ds_A)>0 and len(ds_B)>0
        self.input_nc = 3
        self.output_nc = 3        
        self.A_path = []
        self.B_path = []        
        self.label = []
        if self.opt.generatedroot is not None:
            ds_C = sorted(make_dataset(self.opt.generatedroot, self.opt.max_dataset_size))
            assert len(ds_C)>0
            self.C_path = []
        else:
            ds_C = ds_A

        idx = ds_A[0].split('/').index('NI')
        for imp in ds_A:
            spt = imp.split('/')
            spt[0]='/'
            spt[idx]='VL'
            imp1 = join(*spt)
            if self.opt.generatedroot is None:
                imp2 = imp
            else:
                tmpPath = [self.opt.generatedroot] + spt[idx+2:]
                imp2 = join(*tmpPath)
            if imp1 in ds_B and imp2 in ds_C:
                self.A_path.append(imp)
                self.B_path.append(imp1)
                if self.opt.generatedroot is not None:
                    self.C_path.append(imp2)
                self.label.append(int(spt[idx+2][1:])-1)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_path[index]
        B_path = self.B_path[index]
        label = self.label[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')        

        # apply the same transform to both A and B

        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #transform_params = get_params(self.opt, B.size)
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = self.A_transform(A)
        B = self.B_transform(B)

        if self.opt.generatedroot is None:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'label': label}
        else:
            C_path = self.C_path[index]
            C = Image.open(C_path).convert('RGB')
            C = self.C_transform(C)
            return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.label)

    def reconstruct_A(self, x):
        return self.reconstruct(x, self.denorm_A)

    def reconstruct_B(self, x):
        return self.reconstruct(x, self.denorm_B)

    def reconstruct_C(self, x):
        if self.opt.generatedroot is not None:
            return self.reconstruct(x, self.denorm_C)
        else:
            return x


# For unpaired dataset
# Group A is NIR images
# Group B is VIS images
# Group C is generated images
# Label is the person ID
# It is called unpaired because the image pose, expression may vary a lot even for the same ID
class UnpairedLabelDataset(BaseDataset):
    """unpaired dataset with label
    """

    def __init__(self, opt, seed=100):
        BaseDataset.__init__(self, opt)
        if self.opt.dataset_name == 'casia':
            self.__init_casia()
        else:
            raise NotImplementedError
        random.seed(seed)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, Image.open(self.A_path[0]).convert('RGB').size)
        self.A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_params = get_params(self.opt, Image.open(self.B_path[0]).convert('RGB').size)
        self.B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        self.denorm_A = get_denorm(opt.mean, opt.std, self.input_nc==1) # denorm transformation
        self.denorm_B = get_denorm(opt.mean, opt.std, self.output_nc==1) # denorm transformation

        if self.opt.generatedroot is not None:
            transform_params = get_params(self.opt, Image.open(self.C_path[0]).convert('RGB').size)
            self.C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            self.denorm_C = get_denorm(opt.mean, opt.std, self.output_nc==1) # denorm transformation

    def __init_casia(self):
        dir_A = join(self.root, 'NIR')
        dir_B = join(self.root, 'VIS')
        self.A_path = sorted(make_dataset(dir_A, self.opt.max_dataset_size))
        self.B_path = sorted(make_dataset(dir_B, self.opt.max_dataset_size))
        assert len(self.A_path)>0 and len(self.B_path)>0        
        self.input_nc = 3
        self.output_nc = 3
        self.A_label = []
        self.B_label = []        
        name_list = []
        for p in self.A_path:
            name_list.append(p.split('/')[-1].split('_')[-2])
        for p in self.B_path:
            name_list.append(p.split('/')[-1].split('_')[-2])
        name_list = sorted(list(set(name_list)))
        for p in self.A_path:
            self.A_label.append(name_list.index(p.split('/')[-1].split('_')[-2]))
        for p in self.B_path:
            self.B_label.append(name_list.index(p.split('/')[-1].split('_')[-2]))
        self.A_size = len(self.A_path)
        self.B_size = len(self.B_path)

        if self.opt.generatedroot is not None:
            self.C_path = sorted(make_dataset(self.opt.generatedroot, self.opt.max_dataset_size))
            assert len(self.C_path)>0
            self.C_label = []
            for p in self.C_path:
                self.C_label.append(name_list.index(p.split('/')[-1].split('_')[-2]))
            self.C_size = len(self.C_path)
        else:
            self.C_size = 0

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        index_A = index % self.A_size
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_path[index_A]  # make sure index is within then range
        B_path = self.B_path[index_B]
        A_label = self.A_label[index_A]
        B_label = self.B_label[index_B]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #transform_params = get_params(self.opt, B.size)
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = self.A_transform(A)
        B = self.B_transform(B)

        if self.opt.generatedroot is None:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label, 'B_label': B_label}
        else:
            if self.opt.serial_batches:   # make sure index is within then range
                index_C = index % self.C_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_C = random.randint(0, self.C_size - 1)
            C_path = self.C_path[index_C]
            C_label = self.C_label[index_C]
            C = Image.open(C_path).convert('RGB')
            C = self.C_transform(C)
            return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'A_label': A_label, 'B_label': B_label, 'C_label': C_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size, self.C_size)

    def reconstruct_A(self, x):
        return self.reconstruct(x, self.denorm_A)

    def reconstruct_B(self, x):
        return self.reconstruct(x, self.denorm_B)

    def reconstruct_C(self, x):
        if self.opt.generatedroot is not None:
            return self.reconstruct(x, self.denorm_C)
        else:
            return x

# For mixed dataset
# For generated color images, the label is 0
# For real color images, the label is 1
# For real nir images, the label is 2
# Accept multiple folders
class MixedDataset(BaseDataset):
    """unpaired dataset with label
    """

    def __init__(self, opt, seed=100):
        BaseDataset.__init__(self, opt)
        assert len(opt.dataroot) == len(opt.label)
        self.input_nc = 3
        self.A_path = []
        self.A_label = []
        for i, root in enumerate(opt.dataroot):
            tmp = sorted(make_dataset(root, self.opt.max_dataset_size))
            assert len(tmp)>0
            self.A_path += tmp
            self.A_label += [opt.label[i]] * len(tmp)
        self.A_size = len(self.A_path)

        random.seed(seed)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, Image.open(self.A_path[0]).convert('RGB').size)
        self.A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        self.denorm_A = get_denorm(opt.mean, opt.std, self.input_nc==1) # denorm transformation

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        index_A = index % self.A_size

        A_path = self.A_path[index_A]  # make sure index is within then range
        A_label = self.A_label[index_A]
        A = Image.open(A_path).convert('RGB')

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        A = self.A_transform(A)

        return {'A': A, 'A_paths': A_path, 'A_label': A_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size

    def reconstruct_A(self, x):
        return self.reconstruct(x, self.denorm_A)

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/__init__.py
# CustomDatasetDataLoader()
class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, dataset_name, dataset_opt, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset_opt = dataset_opt
        dataset_class = find_dataset_using_name(opt.data_lib, dataset_name)
        self.dataset = dataset_class(dataset_opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.dataset_opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.dataset_opt.max_dataset_size:
                break
            yield data


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/__init__.py
# find_dataset_using_name()
def find_dataset_using_name(data_lib, dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    datasetlib = importlib.import_module(data_lib)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name == dataset_name \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset