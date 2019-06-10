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
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


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

    def __init_oulu(self):
        dir_A = join(self.root, 'NI/Strong')
        dir_B = join(self.root, 'VL/Strong')
        ds_A = sorted(make_dataset(dir_A, self.opt.max_dataset_size))
        ds_B = sorted(make_dataset(dir_B, self.opt.max_dataset_size))
        assert len(ds_A) > 0 and len(ds_B) > 0
        self.input_nc = 3
        self.output_nc = 3        
        self.A_path = []
        self.B_path = []
        self.label = []
        idx = ds_A[0].split('/').index('NI')
        for imp in ds_A:
            spt = imp.split('/')
            spt[0]='/'
            spt[idx]='VL'
            imp1 = join(*spt)
            if imp1 in ds_B:
                self.A_path.append(imp)
                self.B_path.append(imp1)
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

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_params = get_params(self.opt, B.size)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': A_path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.label)


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
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_params = get_params(self.opt, B.size)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': A_path, 'A_label': A_label, 'B_label': B_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)