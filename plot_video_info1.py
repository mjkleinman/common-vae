from utils.datasets import VideoSprites
import torch
from torch.utils.data import Dataset, DataLoader
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import _kl_normal_loss
import seaborn as sns
import matplotlib.pyplot as plt

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

NAT_TO_BITS = 1.44

if __name__ == '__main__':

    res_list = []
    for frame in range(8):
        dataset = VideoSprites(logger=None, frames=frame)
        pin_memory = torch.cuda.is_available
        dataloader = DataLoader(dataset,
                                batch_size=100,
                                shuffle=False,
                                pin_memory=pin_memory)

        directory = f"results-video/results/vsprites_frames={frame}_z=9_zu=3_epoch=25_klu=10_klqq=1"
        model = load_model(directory, is_gpu=True)

        klc1 = AverageMeter()
        klc2 = AverageMeter()
        model.eval()
        device = next(model.parameters()).device

        for (_, input1, input2), target in dataloader:
            with torch.no_grad():
                xa = input1.to(device=device)
                xb = input2.to(device=device)
                _, _, pmc1, plc1, _, _, pmc2, plc2 = model.encoder(xa, xb)
                klc1_temp = _kl_normal_loss(pmc1, plc1)
                klc2_temp = _kl_normal_loss(pmc2, plc2)

                klc1.update(klc1_temp.item())
                klc2.update(klc2_temp.item())

        res_list.append((klc1.avg + klc2.avg)/2 * NAT_TO_BITS) # 1.44 to make in Bits

    sns.set(font_scale=1.0)
    widths = list(range(8))
    plt.figure(figsize=(4, 3.2))
    plt.plot(widths, res_list, color='blue', marker='o')
    plt.ylabel('Common Information Rate (bits)')
    plt.xlabel('Delay between frames')
    plt.savefig('plots/video_info_oct122023_bits.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


