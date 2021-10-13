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
                 "ddsprites": "DoubleDSprites"}
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
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
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
    dataset = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
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

    def __init__(self, root=os.path.join(DIR, '../data/dsprites/'), **kwargs):
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


class DoubleDSprites(DSprites):
    def __init__(self, **kwargs):
        super(DoubleDSprites, self).__init__()
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
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)
        sample_b = np.expand_dims(self.imgs[idx_b] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample, sample_b = self.transforms(sample), self.transforms(sample_b)
        img_cat = torch.cat((sample, sample_b), dim=0)

        lat_value = self.lat_values[idx]
        return (img_cat, sample, sample_b), lat_value


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


# https://github.com/avivga/lord-pytorch/blob/main/dataset.py
# class Shapes3D(DisentangledDataset):

#     def __init__(self, base_dir):
#         super().__init__(base_dir)

#         self.__data_path = os.path.join(base_dir, '3dshapes.h5')

#     def __img_index(self, floor_hue, wall_hue, object_hue, scale, shape, orientation):
#         return (
#             floor_hue * 10 * 10 * 8 * 4 * 15
#             + wall_hue * 10 * 8 * 4 * 15
#             + object_hue * 8 * 4 * 15
#             + scale * 4 * 15
#             + shape * 15
#             + orientation
#         )

#     def read_images(self):
#         with h5py.File(self.__data_path, 'r') as data:
#             imgs = data['images'][:]
#             classes = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
#             content_ids = dict()

#             for floor_hue in range(10):
#                 for wall_hue in range(10):
#                     for object_hue in range(10):
#                         for scale in range(8):
#                             for shape in range(4):
#                                 for orientation in range(15):
#                                     img_idx = self.__img_index(floor_hue, wall_hue, object_hue, scale, shape, orientation)
#                                     content_id = '_'.join((str(floor_hue), str(wall_hue), str(object_hue), str(scale), str(orientation)))

#                                     classes[img_idx] = shape
#                                     content_ids[img_idx] = content_id

#             unique_content_ids = list(set(content_ids.values()))
#             contents = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
#             for img_idx, content_id in content_ids.items():
#                 contents[img_idx] = unique_content_ids.index(content_id)

#             return imgs, classes, contents

# taken from: https://github.com/mmrl/disent-and-gen/blob/master/src/dataset/shapes3d.py
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

    def __init__(self, imgs=None, latent_values=None, latent_classes=None, color_mode='rgb',
                 transform=None, target_transform=None):

        path = Shapes3D.files['train']
        data_zip = h5py.File(path, 'r')
        self.imgs = data_zip['images'][()]
        self.latent_values = data_zip['labels'][()]
        self.latent_classes = latent_values
        self.latents_sizes = np.array([10, 10, 10, 8, 4, 15])

        image_transforms = [transforms.ToTensor(),
                            transforms.ConvertImageDtype(torch.float32),
                            transforms.Lambda(lambda x: x.flatten())]

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

    files = {"train": "../data/shapes3d/3dshapes.h5"}
    n_gen_factors = 6
    lat_names = ('floor_hue', 'wall_hue', 'object_hue',
                 'scale', 'shape', 'orientation')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])

    img_size = (3, 64, 64)

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


class DoubleShapes3D(Shapes3D):
    def __init__(self, **kwargs):
        super(DoubleShapes3D, self).__init__()
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
        return (img_cat, sample, sample_b), 0

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
    dataset = DoubleShapes3D()
    # Dataset = DoubleCeleb  # CelebA
    # logger = logging.getLogger(__name__)
    # dataset = Dataset(logger=logger)
    pin_memory = torch.cuda.is_available
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            pin_memory=pin_memory)

    # print((dataloader.dataset[2]))
    # # print(dataloader.dataset[97231])

    pdb.set_trace()
    for (_, input1, input2), _ in dataloader:
        # print(input1)
        print(torch.sum(input2))
        import sys
        sys.exit()

# if __name__ == '__main__':
#     dataset = DoubleMNIST()
#     trainloader = torch.utils.data.DataLoader(dataset)
#     print(trainloader)
#     for data, _ in trainloader:
#         print(torch.sum(data[0]))
#         sys.exit()
