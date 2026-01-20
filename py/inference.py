import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MNISTDataset
from model import NeuralNetwork

CACHE_DIR = "./.cache"

batch_size = 64

test_dataset = MNISTDataset("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


model = NeuralNetwork()
print(model)

os.makedirs(CACHE_DIR, exist_ok=True)
model = NeuralNetwork()
model.load_state_dict(torch.load(CACHE_DIR + "/model.pth", weights_only=True))

model.eval()
x, y = test_dataset[0][0], test_dataset[0][1]
plt.imshow(x.squeeze(), cmap="gray")
plt.show()

loss_fn = nn.CrossEntropyLoss()
pred = model(x)
print(pred)
target = torch.tensor([y], dtype=torch.long)

print(pred.shape, target.shape)
loss = loss_fn(pred, target)
print(f"Loss: {loss.item():.4f}")

