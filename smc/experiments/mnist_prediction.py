import numpy as np
import torch
from pkg_resources import resource_filename
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.datasets import MNIST

from smc import DEVICE
from smc.models import MNIST_Net, MNISTRepLearner

torch.manual_seed(41310)
np.random.seed(41310)

DATA_LOC = resource_filename("smc", "data_loading")

model = MNISTRepLearner()
model.load_model(name="test_CR")

test_dataset = MNIST(
    root=DATA_LOC,
    transform=transforms.ToTensor(),
    download=True,
    train=False,
)

test_transform_dataset = MNIST(
    root=DATA_LOC,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    download=True,
    train=False,
)
data_loader = torch.utils.data.DataLoader(  # type: ignore
    test_transform_dataset, batch_size=10000, shuffle=True
)

t_data = list(data_loader)[0][0].to(DEVICE)

predictors = [MNIST_Net(load=(str(digit) + "_new")) for digit in range(10)]

reps = range(10)

n_samples = [3, 5, 7, 10, 20, 50, 100, 500, 1000, 5000]

aucs = np.zeros((len(n_samples), len(reps)))

preds = np.zeros((10000, 10, 10))

for i, predictor in enumerate(predictors):

    pred = predictor.forward(t_data)

    preds[:, i, :] = np.exp(pred.detach().cpu())
    preds[:, i, :] /= preds[:, i, :].sum(axis=1).reshape((10000, 1))


for i, num in enumerate(n_samples):
    for rep in reps:

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

            _, latents, _ = model.forward(dataset.data.float().to(DEVICE))

            latents = latents.detach().cpu().numpy()
            latents = latents[np.random.choice(latents.shape[0], num, replace=False), :]
            kernel = gaussian_kde(latents.T)
            kdes.append(kernel)

        _, test_latents, _ = model.forward(test_dataset.data.float().to(DEVICE))

        weights = np.zeros((10000, 10))

        for digit, (predictor, kernel) in enumerate(zip(predictors, kdes)):

            density = kernel(test_latents.detach().cpu().T)
            weights[:, digit] = density

        weights = weights / weights.sum(axis=1).reshape(-1, 1)

        new_preds = weights.reshape((10000, 10, 1)) * preds
        new_preds = new_preds.sum(axis=2)

        auc = roc_auc_score(
            y_true=test_dataset.targets, y_score=new_preds, multi_class="ovr"
        )

        print(f"{num} samples, rep {rep}, AUC: {auc}")
        aucs[i, rep] = auc

np.save("MNIST_pred_results_CR.npy", aucs)
