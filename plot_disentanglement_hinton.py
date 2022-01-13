import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import seaborn as sns
import argparse
import pdb
import pandas as pd
from disvae.utils.modelIO import load_np_arrays
from utils.helpers import retrieve_object
from analysis.hinton import hinton


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help="Name of the model for storing and loading purposes.")
parser.add_argument('--dataset', type=str)
parser.add_argument('--nu', type=int, help="Number of unique latents")
parser.add_argument('--nz', type=int, help="Number of latents")
parser.add_argument('--num-factors', type=int, help="Number of factors")
args = parser.parse_args()

exp_dir = args.name
exp_dir = 'results/' + exp_dir

vae_scores = retrieve_object('disent_scores.p', path=exp_dir)
vae_R = vae_scores.R_coeff
hinton(vae_R, 'factor', 'latent', fontsize=18, save_plot=True, figs_dir=exp_dir)

# loss_plot = []
# klqq_loss_plot = []
# for frames in range(8):
#     directory = f"results-video/results/vsprites_frames={frames}_z=9_zu=3_epoch=25_klu=10_klqq=1"
#     path_to_metadata = os.path.join(directory, filename)
#     klc_loss, klqq_loss = read_loss_from_file(path_to_metadata, 'klc_loss')
#     loss_plot.append(klc_loss)
#     klqq_loss_plot.append(klqq_loss)

# print(loss_plot)
# print(klqq_loss_plot)

# sns.set(font_scale=1.0)
# widths = list(range(8))
# plt.plot(widths, loss_plot, label='klc', color='blue', marker='o')
# plt.plot(widths, klqq_loss_plot, label='klqq', color='green', marker='o')
# plt.ylabel('Common Information Rate (bits)')
# plt.legend()
# plt.xlabel('Delay between frames')
# plt.savefig(args.plot_path, format='pdf', dpi=None, bbox_inches='tight')
# plt.close()
