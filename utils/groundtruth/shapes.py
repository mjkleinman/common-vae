# taken from: https://github.com/mmrl/disent-and-gen/blob/master/src/dataset/shapes3d.py
import subprocess
import os
import numpy as np
import h5py
import pdb

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode

COLOUR_BLACK = 0
COLOUR_WHITE = 1
DIR = os.path.abspath(os.path.dirname(__file__))


class Shapes3D(Dataset):
    """
    Disentangled dataset used in Kim and Mnih, (2019)

    #==========================================================================
    # Latent Dimension,    Latent values                                 N vals
    #==========================================================================

    # floor hue:           uniform in range [0.0, 1.0)                      10
    # wall hue:            uniform in range [0.0, 1.0)                      10
    # object hue:          uniform in range [0.0, 1.0)                      10
    # scale:               uniform in range [0.75, 1.25]                     8
    # shape:               0=square, 1=cylinder, 2=sphere, 3=pill            4
    # orientation          uniform in range [-30, 30]                       15
    """

    files = {"train": "../../data/shapes3d/3dshapes.h5"}
    n_gen_factors = 6
    lat_names = ('floor_hue', 'wall_hue', 'object_hue',
                 'scale', 'shape', 'orientation')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])

    img_size = (3, 64, 64)

    background_color = COLOUR_WHITE
    lat_values = {'floor_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'wall_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'object_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  'scale': [0.75, 0.82142857, 0.89285714, 0.96428571,
                            1.03571429, 1.10714286, 1.17857143, 1.25],
                  'shape': [0, 1, 2, 3],
                  'orientation': [-30., -25.71428571, -21.42857143,
                                  -17.14285714, -12.85714286, -8.57142857,
                                  -4.28571429, 0., 4.28571429, 8.57142857,
                                  12.85714286, 17.14285714, 21.42857143,
                                  25.71428571, 30.]}

    def __init__(self, imgs=None, latent_values=None, latent_classes=None, color_mode='rgb',
                 transform=None, target_transform=None):

        path = Shapes3D.files['train']
        data_zip = h5py.File(os.path.join(DIR, path), 'r')
        self.imgs = data_zip['images'][()]
        self.latent_values = data_zip['labels'][()]
        self.latent_classes = latent_values
        self.latents_sizes = np.array([10, 10, 10, 8, 4, 15])

        image_transforms = [transforms.ToTensor(),
                            transforms.ConvertImageDtype(torch.float32)]

        if color_mode == 'hsv':
            image_transforms.insert(0, transforms.Lambda(rgb2hsv))

        latent_transforms = [transforms.Lambda(lambda x: torch.from_numpy(x).to(
            dtype=torch.float32))]

        self.transforms = transforms.Compose(image_transforms)
        self.target_transforms = transforms.Compose(latent_transforms)

    def __getitem__(self, key):
        return (self.imgs[key],
                self.imgs[key],
                self.imgs[key]), 0

    def __len__(self):
        return len(self.imgs)


class DoubleShapes3DBase(Shapes3D):
    def __init__(self, **kwargs):
        super(DoubleShapes3DBase, self).__init__()
        # self.latents_sizes = self.dataset_zip['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                             np.array([1, ])))

    # @staticmethod
    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    # @staticmethod
    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        samples_b = np.zeros((size, self.latents_sizes.size))

        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
            samples_b[:, lat_i] = samples[:, lat_i]

        # get second sample
        for lat_i, lat_size in enumerate(self.latents_sizes[:3]):
            samples_b[:, lat_i] = np.random.randint(lat_size, size=size)

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
        sample = self.imgs[idx]
        sample_b = self.imgs[idx_b]

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample, sample_b = self.transforms(sample), self.transforms(sample_b)
        img_cat = torch.cat((sample, sample_b), dim=0)

        # lat_value = self.lat_values[idx]
        return (img_cat, sample, sample_b), latent_idx.astype(int).reshape(-1)


class DoubleShapes3D(DoubleShapes3DBase):
    def __init__(self, **kwargs):
        super(DoubleShapes3D, self).__init__()


class DoubleShapes3DViewUnq(DoubleShapes3DBase):
    def __init__(self, **kwargs):
        super(DoubleShapes3DViewUnq, self).__init__()

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        samples_b = np.zeros((size, self.latents_sizes.size))

        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
            samples_b[:, lat_i] = samples[:, lat_i]

        # get second sample
        for lat_i, lat_size in enumerate(self.latents_sizes):
            if lat_i == 5:
                samples_b[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples, samples_b


class DoubleShapes3DCorrelated(DoubleShapes3DBase):
    def __init__(self, **kwargs):
        super(DoubleShapes3DCorrelated, self).__init__()

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        samples_b = np.zeros((size, self.latents_sizes.size))

        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
            samples_b[:, lat_i] = samples[:, lat_i]

        # get second sample
        for lat_i, lat_size in enumerate(self.latents_sizes[:3]):
            choices = [(lat_i + offset) % lat_size for offset in range(1, 1 + lat_size // 2)]
            samples_b[:, lat_i] = np.random.choice(choices, size=size)

        return samples, samples_b
