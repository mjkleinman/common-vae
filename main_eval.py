from utils.datasets import get_dataloaders, get_img_size, DATASETS
from disvae.utils.modelIO import save_model, load_model, load_metadata
from analysis.metrics import DCIMetrics
from analysis.hinton import hinton
import torch
import os
# from utils.viz_helpers import get_samples_and_targets
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from disvae.utils.modelIO import save_model, load_model, load_metadata
import numpy as np


device = 'cuda'


def infer(model, data):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in data:
            x, xa, xb = x
            x = x.to(device=device)
            xa = xa.to(device=device)
            xb = xb.to(device=device)

            pmu1, plu1, pmc1, plc1, pmu2, plu2, _, _ = model.encoder(xa, xb)
            post_mean = torch.cat((pmu1, pmc1, pmu2), dim=-1)  # just using the mean from xa
            latents.append(post_mean.cpu())
            targets.append(t)
            break

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets


test_loader = get_dataloaders("ddsprites2",
                              batch_size=10000,
                              shuffle=True,
                              logger=None)

# for this experiment, there were 7 latent components.
# first 2 are unique for view1, middle 3 are common, and last 2 are unique for view2
exp_dir = 'cvae_ddsprites2_randSample_klqq=0.1_klu=10_epoch=70_z=7_zu=2'
exp_dir = 'results/' + exp_dir
model = load_model(exp_dir, is_gpu=True)
l, t = infer(model, test_loader)

# Full data
metric_data = (l, t)
eastwood = DCIMetrics(metric_data, n_factors=5, regressor='ensemble')
vae_scores = eastwood(model, model_zs=metric_data)

# Common latents
metric_data_common = (l[:, 2:5], t)
eastwood_common = DCIMetrics(metric_data, n_factors=5, regressor='ensemble')
vae_scores_common = eastwood_common(model, model_zs=metric_data_common)

# Unique latents for View A
metric_data_unique = (l[:, :2], t)
eastwood_unique = DCIMetrics(metric_data, n_factors=5, regressor='ensemble')
vae_scores_unique = eastwood_unique(model, model_zs=metric_data_unique)

# vae_R = vae_scores.R_coeff
# hinton(vae_R, 'factor', 'latent', fontsize=18, save_plot=True, figs_dir=exp_dir)
with open(os.path.join(exp_dir, 'metrics.txt'), 'w') as f:
    f.write('common log losses: ' + str(vae_scores_common.log_losses) + "\n")
    f.write('unique log losses: ' + str(vae_scores_unique.log_losses) + "\n")
    f.write('total log losses: ' + str(vae_scores.log_losses) + "\n")

    f.write('common info: ' + str(1.44 * (np.log([3, 6, 40, 32, 32]) - vae_scores_common.log_losses)) + "\n")
    f.write('unique info: ' + str(1.44 * (np.log([3, 6, 40, 32, 32]) - vae_scores_unique.log_losses)) + "\n")
    f.write('total info: ' + str(1.44 * (np.log([3, 6, 40, 32, 32]) - vae_scores.log_losses)) + "\n")
