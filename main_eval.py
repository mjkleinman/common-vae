from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import infer, store_object
from disvae.utils.modelIO import save_model, load_model, load_metadata
from analysis.metrics import DCIMetrics
from analysis.hinton import hinton
import torch
import os
# from utils.viz_helpers import get_samples_and_targets
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from disvae.utils.modelIO import save_model, load_model, load_metadata
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help="Name of the model for storing and loading purposes.")
parser.add_argument('--dataset', type=str)
parser.add_argument('--nu', type=int, help="Number of unique latents")
parser.add_argument('--nz', type=int, help="Number of latents")
parser.add_argument('--num-factors', type=int, help="Number of factors")
parser.add_argument('--num-samples', type=int,  default=10000, help="Number of samples")
args = parser.parse_args()
device = 'cuda'
if args.dataset.startswith('ddsprites'):  # == 'ddsprites2':
    latent_sizes = [3, 6, 40, 32, 32]
elif args.dataset == 'tmnist':
    latent_sizes = [10, 10]
else:
    latent_sizes = [10, 10, 10, 8, 4, 15]


test_loader = get_dataloaders(args.dataset,
                              batch_size=args.num_samples,
                              shuffle=True,
                              logger=None)

# for this experiment, there were 7 latent components.
# first 2 are unique for view1, middle 3 are common, and last 2 are unique for view2
exp_dir = args.name
exp_dir = 'results-paper/' + exp_dir
model = load_model(exp_dir, is_gpu=True)
l, t = infer(model, test_loader)

# Full data
metric_data = (l, t)
eastwood = DCIMetrics(metric_data, n_factors=args.num_factors, regressor='ensemble')
vae_scores = eastwood(model, model_zs=metric_data)

# Common latents
metric_data_common = (l[:, args.nu:args.nz - args.nu], t)
eastwood_common = DCIMetrics(metric_data, n_factors=args.num_factors, regressor='ensemble')
vae_scores_common = eastwood_common(model, model_zs=metric_data_common)

# Unique latents for View A
metric_data_unique = (l[:, :args.nu], t)
eastwood_unique = DCIMetrics(metric_data, n_factors=args.num_factors, regressor='ensemble')
vae_scores_unique = eastwood_unique(model, model_zs=metric_data_unique)

# Unique latents for View B
metric_data_uniqueB = (l[:, -args.nu:], t)
eastwood_uniqueB = DCIMetrics(metric_data, n_factors=args.num_factors, regressor='ensemble')
vae_scores_uniqueB = eastwood_uniqueB(model, model_zs=metric_data_uniqueB)

# Save files for plotting
store_object(vae_scores, 'disent_scores.p', exp_dir)
store_object(vae_scores_common, 'disent_scores_common.p', exp_dir)
store_object(vae_scores_unique, 'disent_scores_unique.p', exp_dir)
store_object(vae_scores_uniqueB, 'disent_scores_uniqueB.p', exp_dir)


# vae_R = vae_scores.R_coeff
# hinton(vae_R, 'factor', 'latent', fontsize=18, save_plot=True, figs_dir=exp_dir)
with open(os.path.join(exp_dir, 'metrics.txt'), 'w') as f:
    f.write('common log losses: ' + str(vae_scores_common.log_losses) + "\n")
    f.write('unique log losses: ' + str(vae_scores_unique.log_losses) + "\n")
    f.write('unique_B log losses: ' + str(vae_scores_uniqueB.log_losses) + "\n")
    f.write('total log losses: ' + str(vae_scores.log_losses) + "\n")

    f.write('common info: ' + str(1.44 * (np.log(latent_sizes) - vae_scores_common.log_losses)) + "\n")
    f.write('unique info: ' + str(1.44 * (np.log(latent_sizes) - vae_scores_unique.log_losses)) + "\n")
    f.write('unique info B: ' + str(1.44 * (np.log(latent_sizes) - vae_scores_uniqueB.log_losses)) + "\n")
    f.write('total info: ' + str(1.44 * (np.log(latent_sizes) - vae_scores.log_losses)) + "\n")
