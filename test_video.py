from utils.datasets import VideoSprites
import torch
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':

    dataset = VideoSprites()
    pin_memory = torch.cuda.is_available
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            pin_memory=pin_memory)

    for (_, input1, input2), target in dataloader:
        print(target)
        print(torch.sum(input2))
        import sys
        sys.exit()
