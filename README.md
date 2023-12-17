# Gacs-Korner Common Information Variational Autoencoder

Official Pytorch Implementation for [Gacs-Korner Common Information Variational Autoencoder (NeurIPS 2023)](https://arxiv.org/abs/2205.12239) .

## Quick Start

The main training script is `main.py`.

For example, to train on multiple views of dsprites:

`python main.py cvae_ddsprites_randSample_klqq=0.1_klu=10_epoch=70_batch=128_z=8_zu=3_seed=0 -s 0 -d ddsprites -m Doubleburgess -md Doubleburgess -l CVAE --lr 0.001 -b 128 -e 70 -z 8 -zu 3 --gamma-klu 10 --gamma-klqq 0.1`

These commands (and the others used in the paper) are generated in `scripts/generate_paper_commands.py`.

## Evaluating Information and Disentanglement

The main evaluation script is `main_eval.py`. For example, run:

`python main_eval.py --name cvae_ddsprites_randSample_klqq=0.1_klu=10_epoch=70_batch=128_z=8_zu=3_seed=0 --dataset ddsprites --nu 3 --nz 8 --num-factors 5`

Note that `--num-factors 5` corresponds to the number of ground-truth latent factors.

## Generating visualizations

The main visualization script is `main_viz.py`. To generate traversals, run:

`python main_viz.py cvae_ddsprites_randSample_klqq=0.1_klu=10_epoch=70_batch=128_z=8_zu=3_seed=0 traversals -r 8`

Plotting scripts (which show how to load logged data) begin with `plot_*`. For example, run:

`python plot_disentanglement_hinton.py --result-dir results --name  cvae_ddsprites_randSample_klqq=0.1_klu=10_epoch=70_batch=128_z=8_zu=3_seed=0`

Visualizations, plots, and metrics will be in `results/{name}`

## Requirements

- python 3.6+
- torch
- torchvision
- scipy
- seaborn (for plotting)

This repository builds off: https://github.com/YannDubs/disentangling-vae

---
If you find this useful for your work, please consider citing
```
@inproceedings{
kleinman2023gacskorner,
title={Gacs-Korner Common Information Variational Autoencoder},
author={Michael Kleinman and Alessandro Achille and Stefano Soatto and Jonathan Kao},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=e4XidX6AHd}
}
```
