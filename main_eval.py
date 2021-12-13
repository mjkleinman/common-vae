from utils.datasets import get_dataloaders, get_img_size, DATASETS
from disvae.utils.modelIO import save_model, load_model, load_metadata
from analysis.metrics import DCIMetrics
from analysis.hinton import hinton
import torch
import os
from utils.viz_helpers import get_samples_and_targets


device = 'cuda'
# def infer(model, samples_a, samples_b, targ):
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

			# xa = samples_a.to(device=device)
			# xb = samples_b.to(device=device)
			post_mean, post_logvar, _, _, _, _ = model.encoder(xa, xb)
			latents.append(post_mean.cpu())
			targets.append(t[0])

	latents = torch.cat(latents)
	targets = torch.cat(targets)

	return latents, targets

test_loader = get_dataloaders("dshapesd",
                              batch_size=128,
                              shuffle=True,
                              logger=None)

betas = [2, 8, 16]
batchs = [32, 64]

# betas = [2]
# batchs = [32]

# num_samples = 1000
# dataset = 'dshapesd'
# samples, samples_a, samples_b, targ = get_samples_and_targets(dataset, num_samples)

for beta in betas:
	for batch in batchs:
		exp_dir = f'avae_actpost_beta={beta}_klqq=0.0_dshapesd_fb=0_epoch=30_z=10_lr=0.0001_batch={batch}_seed=0'
		exp_dir = 'results/' + exp_dir
		model = load_model(exp_dir, is_gpu=True)
		l, t = infer(model, test_loader)
		metric_data = (l, t)
		eastwood = DCIMetrics(metric_data, n_factors=6)
		vae_scores = eastwood(model, model_zs = metric_data)
		vae_R = vae_scores.R_coeff
		hinton(vae_R, 'factor', 'latent', fontsize=18, save_plot=True, figs_dir=exp_dir)
		with open(os.path.join(exp_dir, 'disentanglement.txt'), 'w') as f:
		    f.write('overall disentanglement: ' + format(vae_scores.overall_disentanglment, '.3f') + "\n")
