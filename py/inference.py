import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader

from dataset import MNISTDataset
from model import NeuralNetwork

CACHE_DIR = "./.cache"

batch_size = 64

test_dataset = MNISTDataset("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

os.makedirs(CACHE_DIR, exist_ok=True)
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(CACHE_DIR + "/model.pth", weights_only=True))

model.eval()
x, y = test_dataset[0][0], test_dataset[0][1]
plt.imshow(x.squeeze(), cmap="gray")
plt.show()

# Dictionary to store results
activations = {}

def hook_fn(module, input, output):
    # Find the name of the module to use as a dictionary key
    name = str(module)
    activations[name] = output.detach().cpu().numpy()

# Attach hooks to every Linear layer found in the model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.register_forward_hook(hook_fn)


with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    print(pred)


# View your results
for layer_name, values in activations.items():
    print(f"Layer: {layer_name} | First 5 values: {values[0][:5]}")

