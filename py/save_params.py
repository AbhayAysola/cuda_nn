import os

from model import NeuralNetwork
import torch

CACHE_DIR = "./.cache"

os.makedirs(CACHE_DIR, exist_ok=True)
model = NeuralNetwork()
model.load_state_dict(torch.load(CACHE_DIR + "/model.pth", weights_only=True))

file = open(CACHE_DIR + "/params.bin", "wb")
params = []
for name, param in model.named_parameters():
    data = param.data.numpy()
    params.append(data)

file.write((len(params)//2).to_bytes(4, "little"))
for param in params:
    print(param.shape)
    if len(param.shape) == 2:
        file.write(param.shape[0].to_bytes(2, "little"))
        file.write(param.shape[1].to_bytes(2, "little"))

for param in params:
    file.write(param.tobytes())
file.close()
print(params[1])
