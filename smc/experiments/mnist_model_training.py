from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn
from pkg_resources import resource_filename
import torch
import numpy as np

from smc.models import MNIST_Net

np.random.seed(41310)
torch.manual_seed(41310)

DATA_LOC = resource_filename("smc", "data_loading")

for digit in range(10):

    dataset = MNIST(
        root=DATA_LOC,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        download=True,
    )

    idx = dataset.targets == digit
    for i in range(len(idx)):
        if (i % 10) == digit:
            idx[i] = True

    print(idx.sum())

    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    model = MNIST_Net()

    model.fit(dataset, epochs=10)
    model.average_input = nn.Parameter(dataset.data.float().mean(0))
    model.save_model(name=(str(digit) + "_new"))
