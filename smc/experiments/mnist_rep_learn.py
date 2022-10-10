from smc.models import MNIST_Net, MNISTRepLearner
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import numpy as np
from pkg_resources import resource_filename

np.random.seed(41310)
torch.manual_seed(41310)

DATA_LOC = resource_filename("smc", "data_loading")

model = MNISTRepLearner()

dataset = MNIST(
    root=DATA_LOC,
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
model.fit(dataset, epochs=50, learning_rate=1e-3)
model.save_model(name="test_CR")
