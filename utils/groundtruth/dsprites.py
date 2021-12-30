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

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1

# from datasets import DisentangledDataset


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


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, root=os.path.join(DIR, '../../data/dsprites/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']
        self.latents_sizes = np.array([3, 6, 40, 32, 32])

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value


class DoubleDSpritesBase(DSprites):
    def __init__(self, **kwargs):
        """
        isPosUnique: boolean to decide which latents are unq/common. If true, the x,y position belong to the unique components
        """

        super(DoubleDSpritesBase, self).__init__()
        # self.latents_sizes = self.dataset_zip['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                             np.array([1, ])))

    # @staticmethod
    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    # @staticmethod
    def sample_latent(self, size=1):
        samples = np.zeros((size, len(self.latents_sizes)))
        samples_b = np.zeros((size, len(self.latents_sizes)))

        # TODO: Clean up code to make it easier to specify what the common and the unique components are
        # Here latents_sizes is a 1x5 array, where the last two components correspond to the x and y position
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
            samples_b[:, lat_i] = samples[:, lat_i]

        for lat_i, lat_size in enumerate(self.latents_sizes):
            # if not isPosUnique and lat_i < 3:  # pos is common
            if lat_i < 3:
                samples_b[:, lat_i] = np.random.randint(lat_size, size=size)
            # elif isPosUnique and lat_i >= 3:  # pos is unique
            #     samples_b[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples, samples_b

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """

        # todo: set random seed here?
        latent_idx, latent_idx_b = self.sample_latent()
        idx = self.latent_to_index(latent_idx).item()
        idx_b = self.latent_to_index(latent_idx_b).item()

        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)
        sample_b = np.expand_dims(self.imgs[idx_b] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample, sample_b = self.transforms(sample), self.transforms(sample_b)
        img_cat = torch.cat((sample, sample_b), dim=0)

        lat_value = self.lat_values[idx]
        # print(latent_idx)
        # latent_idx = latent_idx.view(-1)
        return (img_cat, sample, sample_b), latent_idx.astype(int).reshape(-1)  # lat_value


# So that naming works with earlier implementations
# This inherits from the base class
class DoubleDSprites(DoubleDSpritesBase):
    def __init__(self, **kwargs):
        super(DoubleDSprites, self).__init__()


# Making the x,y position Unique
# This inherits from the base class
class DoubleDSpritesPosUnique(DoubleDSpritesBase):
    def __init__(self, **kwargs):
        super(DoubleDSpritesPosUnique, self).__init__()

    # @staticmethod
    def sample_latent(self, size=1):
        samples = np.zeros((size, len(self.latents_sizes)))
        samples_b = np.zeros((size, len(self.latents_sizes)))

        # TODO: Clean up code to make it easier to specify what the common and the unique components are
        # Here latents_sizes is a 1x5 array, where the last two components correspond to the x and y position
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
            samples_b[:, lat_i] = samples[:, lat_i]

        # resample second sample
        for lat_i, lat_size in enumerate(self.latents_sizes):
            if lat_i >= 3:  # pos is unique
                samples_b[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples, samples_b
