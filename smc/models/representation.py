import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .mnist_classifier import MNIST_Net


class RepLearner(BaseModel):
    def __init__(self, x_dim):
        super(RepLearner, self).__init__()

        self.name = "RepLearner"

        self.x_dim = x_dim

        self.predictors = [None]

    def encoder(self):
        return

    def decoder(self):
        return

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))  # type: ignore  # pylint: disable=E
        z = self.sample(mu, log_var)
        return self.decoder(z), mu, log_var  # type: ignore  # pylint: disable=too-many-function-args

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def sample_model_dataset(self):

        for predictor in self.predictors:
            sample = predictor.sample()  # type: ignore  # pylint: disable=unused-variable  # noqa

        model_dataset = None

        return model_dataset

    def rec_loss(self, data, data_recon):
        return F.binary_cross_entropy(
            data_recon, data.view(-1, self.x_dim), reduction="sum"
        )

    def kl_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def sep_loss(
        self, model_dataset, pairwise_distances
    ):  # pylint: disable=unused-argument
        return

    def con_loss(
        self, model_dataset_predictions, pairwise_distances
    ):  # pylint: disable=unused-argument
        return

    def loss(self, batch):
        data, targets = batch  # pylint: disable=unused-variable
        data_recon, mu, log_var = self.forward(data)

        model_dataset = self.sample_model_dataset()

        _, latent_models, _ = self.forward(model_dataset)
        pairwise_distances = torch.cdist(latent_models, latent_models)

        model_dataset_predictions = self.get_model_predictions(model_dataset)  # type: ignore

        return (
            self.rec_loss(data, data_recon)
            + 1.0 * self.kl_loss(mu, log_var)
            + 0.1 * self.sep_loss(model_dataset, pairwise_distances)  # type: ignore
            + 0.01 * self.con_loss(model_dataset_predictions, pairwise_distances)  # type: ignore
        )


class MNISTRepLearner(RepLearner):
    def __init__(self):
        super(MNISTRepLearner, self).__init__(x_dim=784)

        self.name = "MNIST_RepLearner"

        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, 2)
        self.fc32 = nn.Linear(256, 2)
        # Decoder
        self.fc4 = nn.Linear(2, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.predictors = [MNIST_Net(load=str(digit)) for digit in range(10)]

    def encoder(self, x):  # pylint: disable=arguments-differ
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def decoder(self, z):  # pylint: disable=arguments-differ
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def sample_model_dataset(self):

        model_dataset = torch.zeros(10, 28, 28)
        for i, predictor in enumerate(self.predictors):
            sample = predictor.average_input.detach()
            model_dataset[i] = sample

        return model_dataset

    def get_model_predictions(self, model_dataset):
        model_preds = torch.zeros(10, 10, 10)
        for i, predictor in enumerate(self.predictors):
            pred = predictor.forward(model_dataset.unsqueeze(1))
            model_preds[i] = pred

        return model_preds

    def sep_loss(self, model_dataset, pairwise_distances):
        return -0.5 * pairwise_distances.mean()

    def con_loss(self, model_dataset_predictions, pairwise_distances):

        # model_dataset_predictions [10,10,10]
        # pairwise_distances [10,10]
        pdist = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=True)

        model_dist = pdist(model_dataset_predictions, model_dataset_predictions)
        # model_dist [10,10,1]

        model_similarity = 1.0 - (
            model_dist.squeeze() / np.sqrt(10.0)
        )  # sqrt(10) max distance

        return (model_similarity * pairwise_distances).sum()
