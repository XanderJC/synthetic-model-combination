from smc.models import MNIST_Net, MNISTRepLearner
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from pkg_resources import resource_filename
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.neighbors import KernelDensity


DATA_LOC = resource_filename("smc", "data_loading")

model = MNISTRepLearner()
model.load_model(name="test")

predictors = [MNIST_Net(load=str(digit)) for digit in range(10)]
kdes = []

for digit in range(10):

    dataset = MNIST(
        root=DATA_LOC,
        transform=transforms.ToTensor(),
        download=True,
    )
    idx = dataset.targets == digit
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    latents = dataset.data.reshape((-1, 784)).detach().numpy()
    # latents = latents[np.random.choice(latents.shape[0], 8, replace=False), :]
    # latents = np.c_[latents, np.ones(len(latents)) * digit]
    print(latents.shape)
    # print(latents)
    # kernel = gaussian_kde(latents.T)
    kernel = KernelDensity(bandwidth=10.0, kernel="gaussian")
    kernel.fit(latents)

    kdes.append(kernel)

test_dataset = MNIST(
    root=DATA_LOC,
    transform=transforms.ToTensor(),
    download=True,
    train=False,
)

test_latents = test_dataset.data.reshape((-1, 784))

preds = np.zeros((10000, 10))

for digit, (predictor, kernel) in enumerate(zip(predictors, kdes)):

    density = np.exp(kernel.score_samples(test_latents.detach()))
    preds[:, digit] = density
    print(digit)

print((preds.sum(axis=1) != 0).mean())
preds = preds / preds.sum(axis=1).reshape(-1, 1)


auc = roc_auc_score(y_true=test_dataset.targets, y_score=preds, multi_class="ovr")

print(auc)
