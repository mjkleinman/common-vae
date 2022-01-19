import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np
import h5py
import pdb

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode

from utils.groundtruth.shapes import DoubleShapes3D, DoubleShapes3DViewUnq
from utils.groundtruth.dsprites import DoubleDSprites, DoubleDSpritesPosUnique
from data.Sprites.load_sprites import sprites_act

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "dsprites": "DSprites",
                 "celeba": "CelebA",
                 "chairs": "Chairs",
                 "dmnist": "DoubleMNIST",
                 "tmnist": "TangleMNIST",
                 "rmnist": "DoubleRotateMNIST",
                 "dceleba": "DoubleCelebA",
                 "pceleba": "PairCelebA",
                 "ddsprites": "DoubleDSprites",
                 "ddsprites2": "DoubleDSpritesPosUnique",
                 "dshapes": "DoubleShapes3D",
                 "dshapes2": "DoubleShapes3DViewUnq",
                 "vsprites": "VideoSprites"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), frames=None, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    if frames is not None:
        dataset = Dataset(logger=logger, frames=frames)
    else:
        dataset = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      num_workers=4,
                      **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class CelebA(DisentangledDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/celeba'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).img_size[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class DoubleCelebA(CelebA):
    def __init__(self, **kwargs):
        super(DoubleCelebA, self).__init__()

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img, img_a, img_b = self.transforms(img), self.transforms(img), self.transforms(img)
        img_a[..., :32] = 0.
        img_b[..., 32:] = 0.

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return (img, img_a, img_b), 0


# https://github.com/mickaelChen/GMV/blob/master/mathieu.py
class PairCelebA(CelebA):
    def __init__(self, dataPath=os.path.join(DIR, '../data/celeba/img_align_celeba'),
                 labelFile=os.path.join(DIR, "../data/celeba/identity_CelebA.txt"), **kwargs):
        super(PairCelebA, self).__init__()
        self.dataPath = dataPath
        with open(labelFile, 'r') as f:
            lines = np.array([p.split() for p in f.readlines()])
        self.files = lines[:, 0]
        self.labels = lines[:, 1].astype(int)
        # self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file1 = self.files[idx]
        file2 = np.random.choice(self.files[self.labels == label])
        img1 = self.transforms(Image.open(os.path.join(self.dataPath, file1)))
        img2 = self.transforms(Image.open(os.path.join(self.dataPath, file2)))
        img_cat = torch.cat((img1, img2), dim=0)
        return (img_cat, img1, img2), 0  # update so output is a concatednated vector of both (or randomly do)


class Chairs(datasets.ImageFolder):
    """Chairs Dataset from [1].

    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).

    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/chairs'),
                 logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.logger.info("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.logger.info("Preprocessing Chairs ...")
        preprocess(os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
                   size=type(self).img_size[1:],
                   center_crop=(400, 400))


class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))

# This will be like the DoubleCifar that I have previously implemented, make sure this works, need to double check that the transforms are working properly here


class DoubleMNIST(MNIST):
    def __init__(self, **kwargs):
        super(DoubleMNIST, self).__init__()

    def __getitem__(self, index):
        img, _ = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        img_a, img_b = img.clone(), img.clone()
        length_image = 16
        img_a[:, :, :length_image] = 0.
        img_b[:, :, length_image:] = 0.
        return (img, img_a, img_b), 0


class TangleMNIST(MNIST):
    def __init__(self, **kwargs):
        super().__init__()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        idx_img2 = np.random.choice(np.flatnonzero(self.targets == target))
        img_b = self.data[idx_img2]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        x_a = Image.fromarray(img.numpy(), mode='L')
        x_b = Image.fromarray(img_b.numpy(), mode='L')

        # taken from https://github.com/jameschapman19/cca_zoo/blob/main/cca_zoo/data/toy.py
        # get random angles of rotation
        rot_a, rot_b = torch.rand(2) * 90 - 45
        x_a_rotate = transforms.functional.rotate(x_a, rot_a.item(), interpolation=InterpolationMode.BILINEAR)
        x_b_rotate = transforms.functional.rotate(x_b, rot_b.item(), interpolation=InterpolationMode.BILINEAR)
        # convert images to tensors
        x_a_rotate = self.transform(x_a_rotate)
        x_b_rotate = self.transform(x_b_rotate)
        img_cat = torch.cat((x_a_rotate, x_b_rotate), dim=0)
        return (img_cat, x_a_rotate, x_b_rotate), 0


class DoubleRotateMNIST(MNIST):
    def __init__(self, **kwargs):
        super().__init__()

    def __getitem__(self, index):
        img = self.data[index]
        img_b = img.clone()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        x_a = Image.fromarray(img.numpy(), mode='L')
        x_b = Image.fromarray(img_b.numpy(), mode='L')

        # get random fixed angles of rotation to apply to x_a and x_b
        rot_a = torch.tensor(30.0)
        rot_b = torch.tensor(-30.0)
        x_a_rotate = transforms.functional.rotate(x_a, rot_a.item(), interpolation=InterpolationMode.BILINEAR)
        x_b_rotate = transforms.functional.rotate(x_b, rot_b.item(), interpolation=InterpolationMode.BILINEAR)
        # convert images to tensors
        x_a_rotate = self.transform(x_a_rotate)
        x_b_rotate = self.transform(x_b_rotate)
        img_cat = torch.cat((x_a_rotate, x_b_rotate), dim=0)
        return (img_cat, x_a_rotate, x_b_rotate), 0


class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/fashionMnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


class PairVideoDataset(datasets.HMDB51):
    def __init__(self, train_tfms, num_frames=16, clip_steps=50, train=True):
        super().__init__('data/hmdb/video_data/', 'data/hmdb/test_train_splits/', num_frames,
                         step_between_clips=clip_steps, fold=1, train=train,
                         transform=train_tfms)

    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/hmdb51.html
    def __getitem__(self, idx):
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        # taking the first and second frame for now
        return video[:, 0, :, :], video[:, 1, :, :]  # , audio, class_index

    def __len__(self):
        return self.video_clips.num_clips()


class VideoSprites(Dataset):
    img_size = (3, 64, 64)  # need to define the image size to be compatible
    background_color = COLOUR_BLACK

    def __init__(self, logger, frames):
        X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('data/Sprites/', return_labels=True)
        # Here X_train contains the video frames, represented as an numpy.array with shape (N_train, T, width, height, N_channel)
        self.X_train = torch.from_numpy(X_train).permute(0, 1, 4, 2, 3)
        self.logger = logger
        self.delta_frames = frames

    def __getitem__(self, idx):
         # taking the first and last frame for now
        img1 = self.X_train[idx, 0, ...]
        img2 = self.X_train[idx, self.delta_frames, ...]
        img_cat = torch.cat((img1, img2), dim=0)
        return (img_cat, img1, img2), 0

    def __len__(self):
        return len(self.X_train)


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


if __name__ == '__main__':

    dataPath = ""
    # dataset = DoubleRotateMNIST()
    # dataset = DoubleShapes3D()
    # dataset = DoubleDSprites()
    dataset = VideoSprites()
    pin_memory = torch.cuda.is_available
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            pin_memory=pin_memory)

    pdb.set_trace()
    for (_, input1, input2), target in dataloader:
        print(target)
        print(torch.sum(input2))
        import sys
        sys.exit()
