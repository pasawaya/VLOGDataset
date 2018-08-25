
from dataset import VLOGDataset

dataset = VLOGDataset(fps=None)

for i in range(1, 2):
    video = dataset[i]
