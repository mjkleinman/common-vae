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
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help="Name of the model for storing and loading purposes.")
parser.add_argument('--result-dir', default='results', type=str, help="Name of the model for storing and loading purposes.")
parser.add_argument('--dataset', type=str)
parser.add_argument('--nu', type=int, help="Number of unique latents")
parser.add_argument('--nz', type=int, help="Number of latents")
parser.add_argument('--num-factors', type=int, help="Number of factors")
args = parser.parse_args()

seeds = ['0', '1', '2']
disent_scores = []

for seed in seeds:
    exp_dir = args.name + seed
    exp_dir = os.path.join(args.result_dir, exp_dir)
    vae_scores = retrieve_object('disent_scores.p', path=exp_dir)
    print(vae_scores.R_coeff[0, :])
    print(vae_scores.R_coeff[6, :])
    print('-----------')

    disent_scores.append(vae_scores.overall_disentanglment)

print(f"mean: {np.mean(disent_scores)}")
print(f"std: {np.std(disent_scores)}")
