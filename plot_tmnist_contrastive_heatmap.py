import numpy as np
import matplotlib
import matplotlib.pyplot as plt

CONTRASTIVE_VALS = [3.285, 0.014] # Taken from CMC/logs-neurips (SEE README_NEURIPS on EC2 instance aws)
FILENAME = 'results-paper/cvae_tmnist_randSample_klqq=0.1_epoch=50_batch=128_z=8_zu=2_seed=0/metrics.txt'

def get_common_vals_from_file(filename):
    with open(filename) as file:
        for line_number, line in enumerate(file):
            if 'total info' in line:
                data = line.split(":")[-1].strip()[1:-1]
                data = data.split() # a list of strings
                data = list(map(lambda x: float(x), data))

    return data

COMMON_VALS = get_common_vals_from_file(FILENAME)


latent = ["label", "rot."]
info_type = ["Contrastive", "GKVAE"]
values = np.array([CONTRASTIVE_VALS,
                    COMMON_VALS])

fig, ax = plt.subplots()
im = ax.imshow(values, cmap = 'Greens')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(latent)))
ax.set_yticks(np.arange(len(info_type)))
ax.set_xticklabels(latent, size=40)
ax.set_yticklabels(info_type, size=40)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(latent)):
    for j in range(len(latent)):
        if j == 1:
            color = "black"
        elif j == 0:
            color = 'w'
        text = ax.text(j, i, round(values[i, j], 3), #rounds to 3 decimal places
                   ha="center", va="center", color=color, size=30)

ax.set_title("Info. in latent (bits)", size=50)
plt.savefig('plots/common_contrastive_heatmap.pdf', format='pdf', dpi=None, bbox_inches='tight')
