import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MNISTDataset
from model import NeuralNetwork, train, test

CACHE_DIR = "./.cache"

batch_size = 32 

train_dataset = MNISTDataset("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = MNISTDataset("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")

os.makedirs(CACHE_DIR, exist_ok=True)
torch.save(model.state_dict(), CACHE_DIR + "/model.pth")
print("Saved PyTorch Model State to model.pth")
