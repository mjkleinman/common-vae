import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import seaborn as sns
import argparse
import pdb
import pandas as pd
from disvae.utils.modelIO import load_np_arrays

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='plot_path', default='plots/video_common_info.pdf', type=str)
args = parser.parse_args()

filename = 'train_losses.log'


# TODO CLEAN THIS UP, THIS IS HARDCODED
def read_loss_from_file(log_file_path, loss_to_fetch):
    logs = pd.read_csv(log_file_path)
    loss = logs['Value'].iloc[-2] / 2  # klc_loss
    klqq_loss = logs['Value'].iloc[-1] / 2  # klqq_loss, check if this should be divided by 2
    return loss, klqq_loss


loss_plot = []
klqq_loss_plot = []
for frames in range(8):
    directory = f"results-video/results/vsprites_frames={frames}_z=9_zu=3_epoch=25_klu=10_klqq=1"
    path_to_metadata = os.path.join(directory, filename)
    klc_loss, klqq_loss = read_loss_from_file(path_to_metadata, 'klc_loss')
    loss_plot.append(klc_loss)
    klqq_loss_plot.append(klqq_loss)

print(loss_plot)
print(klqq_loss_plot)

sns.set(font_scale=1.0)
widths = list(range(8))
plt.plot(widths, loss_plot, color='blue', marker='o')
# plt.plot(widths, klqq_loss_plot, label='klqq', color='green', marker='o')
plt.ylabel('Common Information Rate (bits)')
# plt.legend()
plt.xlabel('Delay between frames')
plt.savefig(args.plot_path, format='pdf', dpi=None, bbox_inches='tight')
plt.close()
