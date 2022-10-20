import numpy as np
import torch
from pkg_resources import resource_filename
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.datasets import MNIST

from smc.models import MNIST_Net, MNISTRepLearner

torch.manual_seed(41310)
np.random.seed(41310)

DATA_LOC = resource_filename("smc", "data_loading")

model = MNISTRepLearner()
model.load_model(name="pretrained")

test_dataset = MNIST(
    root=DATA_LOC,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
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

t_data = list(data_loader)[0][0]

predictors = [MNIST_Net(load=(str(digit) + "_new")) for digit in range(10)]

reps = range(10)


preds = np.zeros((10000, 10, 10))

for i, predictor in enumerate(predictors):

    pred = predictor.forward(t_data)

    preds[:, i, :] = np.exp(pred.detach())
    preds[:, i, :] /= preds[:, i, :].sum(axis=1).reshape((10000, 1))

preds_dist = torch.distributions.categorical.Categorical(torch.tensor(preds))
weights = torch.softmax(1 / preds_dist.entropy(), 1)

new_preds = weights.reshape((10000, 10, 1)) * preds
ent_preds = new_preds.sum(axis=2)  # type: ignore
mv_preds = preds.mean(axis=1)

num = 10
entropy_weighted = np.zeros((num))
majority_voting = np.zeros((num))

for i in range(num):
    indx = slice(i, 10000, num)
    mask2 = np.full(10000, True)
    mask2[indx] = False
    indx = mask2

    auc = roc_auc_score(
        y_true=test_dataset.targets[indx], y_score=ent_preds[indx], multi_class="ovr"
    )

    entropy_weighted[i] = auc

    auc = roc_auc_score(
        y_true=test_dataset.targets[indx], y_score=mv_preds[indx], multi_class="ovr"
    )

    majority_voting[i] = auc

print("Entropy weighted:")
print(
    f"$${(entropy_weighted.mean()).round(1)}"
    "\pm "  # type: ignore  # noqa  # pylint: disable=anomalous-backslash-in-string
    f"{(entropy_weighted.std()).round(1)}$$"
)

print("Majority Voting:")
print(
    f"$${(majority_voting.mean()).round(1)}"
    "\pm"  # type: ignore  # noqa  # pylint: disable=anomalous-backslash-in-string
    f"{(majority_voting.std()).round(1)}$$"
)
