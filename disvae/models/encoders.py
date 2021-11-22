"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn
import pdb
from torch.nn.parameter import Parameter

# ALL encoders should be called Enccoder<Model>


def get_encoder(model_type):
    # model_type = model_type.lower().capitalize()
    if model_type is not "Burgess":
        model_type = "Action"  # hack for now
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class EncoderDoubleburgess(nn.Module):
    def __init__(self, img_size, latent_dim, latent_dim_common):
        super(EncoderDoubleburgess, self).__init__()
        self.latent_dim_common = latent_dim_common
        self.latent_dim_unique = (latent_dim - self.latent_dim_common) // 2  # ASSERT THIS IS AN INTEGER FOR THIS TO WORK
        self.latent_dim_encoder = self.latent_dim_unique + self.latent_dim_common
        self.encoder1 = get_encoder("Burgess")(img_size, self.latent_dim_encoder)
        self.encoder2 = get_encoder('Burgess')(img_size, self.latent_dim_encoder)

    def forward(self, x_a, x_b):
        mu1, logvar1 = self.encoder1(x_a)
        mu2, logvar2 = self.encoder2(x_b)
        # todo: can clean up by splititng in a function
        mu_u1, mu_c1 = mu1[:, :self.latent_dim_unique], mu1[:, self.latent_dim_unique:]
        mu_u2, mu_c2 = mu2[:, :self.latent_dim_unique], mu2[:, self.latent_dim_unique:]
        logvar_u2, logvar_c2 = logvar2[:, :self.latent_dim_unique], logvar2[:, self.latent_dim_unique:]
        logvar_u1, logvar_c1 = logvar1[:, :self.latent_dim_unique], logvar1[:, self.latent_dim_unique:]

        return mu_u1, logvar_u1, mu_c1, logvar_c1, mu_u2, logvar_u2, mu_c2, logvar_c2


class EncoderAction(nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        # self.latent_dim_common = latent_dim_common
        # self.latent_dim_unique = (latent_dim - self.latent_dim_common) // 2  # ASSERT THIS IS AN INTEGER FOR THIS TO WORK
        # self.latent_dim_encoder = self.latent_dim_unique + self.latent_dim_common
        self.encoder = get_encoder("Burgess")(img_size, latent_dim)
        #self.lin1 = Parameter(torch.ones(latent_dim,))  # nn.Linear(latent_dim_common, latent_dim_common, bias=False)
        self.lin1 = nn.Linear(5, latent_dim, bias=False)
        # self.encoder2 = get_encoder('Burgess')(img_size, self.latent_dim_encoder)

    def forward(self, x_a, x_b, action=0.0):
        # action should be a one-hot encoding

        mu1, logvar1 = self.encoder(x_a)
        if self.training:
            mu1_post = mu1 + self.lin1(action.float())  # torch.diag_embed make diagonal  # diagonal weight update, test this
        else:
            mu1_post = mu1
        mu2, logvar2 = self.encoder(x_b)
        # print(self.lin1)
        return mu1, logvar1, mu2, logvar2, mu1_post

# if __name__ == '__main__':
#     encoder = get_encoder('DoubleBurgess')
#     print(encoder(img_size=(1, 32, 32)))
