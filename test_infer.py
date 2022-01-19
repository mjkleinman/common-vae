from utils.helpers import infer, store_object
from analysis.metrics import DCIMetrics
from analysis.hinton import hinton
import torch
import os
# from utils.viz_helpers import get_samples_and_targets
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from disvae.utils.modelIO import save_model, load_model, load_metadata
import numpy as np
import argparse
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help="Name of the model for storing and loading purposes.")
parser.add_argument('--dataset', type=str)
parser.add_argument('--nu', type=int, help="Number of unique latents")
parser.add_argument('--nz', type=int, help="Number of latents")
parser.add_argument('--num-factors', type=int, help="Number of factors")
args = parser.parse_args()
device = 'cuda'
if args.dataset == 'ddsprites2':
    latent_sizes = [3, 6, 40, 32, 32]
else:
    latent_sizes = [10, 10, 10, 8, 4, 15]


test_loader = get_dataloaders(args.dataset,
                              batch_size=30,
                              shuffle=True,
                              logger=None)

# for this experiment, there were 7 latent components.
# first 2 are unique for view1, middle 3 are common, and last 2 are unique for view2
exp_dir = args.name
exp_dir = 'results-paper/' + exp_dir
model = load_model(exp_dir, is_gpu=True)
l, t = infer(model, test_loader)

print(t)
pdb.set_trace()
