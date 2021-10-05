"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder
import random

MODELS = ["Burgess", "Doubleburgess"]


def init_specific_model(model_type_enc, model_type_dec, img_size, latent_dim, latent_dim_unq):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type_enc = model_type_enc.lower().capitalize()
    model_type_dec = model_type_dec.lower().capitalize()
    if model_type_enc not in MODELS or model_type_dec not in MODELS:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type_enc + " " + model_type_dec, MODELS))  # should print decoder model type too

    encoder = get_encoder(model_type_enc)
    decoder = get_decoder(model_type_dec)
    model = DoubleVAE(img_size, encoder, decoder, latent_dim, latent_dim_unq)  # changed to Double
    model.model_type_enc = model_type_enc  # store to help reloading
    model.model_type_dec = model_type_dec
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, latent_dim_unq):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.latent_dim_unq = latent_dim_unq
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim, self.latent_dim - 2 * latent_dim_unq)  # e = c + u
        self.decoder = decoder(img_size, self.latent_dim)  # z = 2u + c, so z = e + u

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class DoubleVAE(VAE):
    def __init__(self, img_size, encoder, decoder, latent_dim, latent_dim_unq):
        super().__init__(img_size, encoder, decoder, latent_dim, latent_dim_unq)

    def reparameterize_double(self, mean_u1, logvar_u1, mean_c1, logvar_c1, mean_u2, logvar_u2, mean_c2, logvar_c2):

        mean_c = 0.5 * (mean_c1 + mean_c2)
        logvar_c = 0.5 * (logvar_c1 + logvar_c2)

        mean_a = torch.cat((mean_u1, mean_c1, mean_u2), dim=-1)
        mean_b = torch.cat((mean_u1, mean_c2, mean_u2), dim=-1)
        logvar_a = torch.cat((logvar_u1, logvar_c1, logvar_u2), dim=-1)
        logvar_b = torch.cat((logvar_u1, logvar_c2, logvar_u2), dim=-1)
        mean = torch.cat((mean_u1, mean_c, mean_u2), dim=-1)
        logvar = torch.cat((logvar_u1, logvar_c, logvar_u2), dim=-1)

        sample1 = self.reparameterize(mean_a, logvar_a)
        sample2 = self.reparameterize(mean_b, logvar_b)
        sample = self.reparameterize(mean, logvar)

        # might be better to concatenate and use both samples or to take the average of the means
        rand_num = random.random()
        if rand_num > 0.5:
            return sample1
        else:
            return sample2
        # else:
        #     return sample

        # this might not work b/c of the different batch size

        # sample = torch.mean((sample1, sample2), dim=0)
        # return sample

    def forward(self, x_a, x_b):
        latent_dists_double = self.encoder(x_a, x_b)
        latent_sample = self.reparameterize_double(*latent_dists_double)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dists_double, latent_sample

    def sample_latent(self, x_a, x_b):
        latent_dists = self.encoder(x_a, x_b)
        latent_sample = self.reparameterize_double(*latent_dists)
        return latent_sample
