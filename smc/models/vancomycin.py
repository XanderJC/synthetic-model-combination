import numpy as np
from smc.models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.stats import multivariate_normal, gaussian_kde


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
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sample(mu, log_var)
        return self.decoder(z), mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def sample_model_dataset(self):

        for predictor in self.predictors:
            sample = predictor.sample()

        model_dataset = None

        return model_dataset

    def rec_loss(self, data, data_recon):
        return F.binary_cross_entropy(
            data_recon, data.view(-1, self.x_dim), reduction="sum"
        )

    def kl_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def sep_loss(self, model_dataset, pairwise_distances):
        return

    def con_loss(self, model_dataset, pairwise_distances):
        return

    def loss(self, batch):
        data, targets = batch
        data_recon, mu, log_var = self.forward(data)

        model_dataset = self.sample_model_dataset()

        _, latent_models, _ = self.forward(model_dataset)
        pairwise_distances = torch.cdist(latent_models, latent_models)

        return (
            self.rec_loss(data, data_recon)
            + 1.0 * self.kl_loss(mu, log_var)
            + 0.1 * self.sep_loss(model_dataset, pairwise_distances)
            + self.con_loss(model_dataset, pairwise_distances)
        )


class VancRepLearner(RepLearner):
    def __init__(self, covariates, pred_matrix, list_densities, mask=None):
        super(VancRepLearner, self).__init__(x_dim=4)

        self.name = "VancRepLearner"

        # Encoder
        self.fc1 = nn.Linear(4, 8)
        self.fc31 = nn.Linear(8, 2)
        self.fc32 = nn.Linear(8, 2)
        # Decoder
        self.fc4 = nn.Linear(2, 8)
        self.fc5 = nn.Linear(8, 4)

        self.predictors = None
        self.covariates = (torch.tensor(covariates)).float()
        self.pred_matrix = (torch.tensor(pred_matrix)).float()
        self.list_densities = list_densities
        self.mask = mask

    def encoder(self, x):
        h = torch.sigmoid(self.fc1(x))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def decoder(self, z):
        h = torch.sigmoid(self.fc4(z))
        return self.fc5(h)

    def sample_model_dataset(self):

        model_dataset = np.zeros((6, 4))
        for i, density in enumerate(self.list_densities):
            sample = density.sample(1)
            model_dataset[i] = sample

        return torch.tensor(model_dataset).float()

    def rec_loss(self, data, data_recon):
        return ((data - data_recon) ** 2).sum()

    def sep_loss(self, model_dataset, pairwise_distances):
        return -0.5 * pairwise_distances.mean()

    def con_loss(self, *args):
        return 0.0

    def loss(self, rho_rec=1.0, rho_kl=1.0, rho_sep=0.0, rho_con=0.0):

        data_recon, mu, log_var = self.forward(self.covariates[:, 0, :4])

        model_dataset = self.sample_model_dataset()

        _, latent_models, _ = self.forward(model_dataset)
        pairwise_distances = torch.cdist(latent_models, latent_models)

        return (
            rho_rec * self.rec_loss(self.covariates[:, 0, :4], data_recon)
            + rho_kl * self.kl_loss(mu, log_var)
            + rho_sep * self.sep_loss(model_dataset, pairwise_distances)
            + rho_con * self.con_loss(model_dataset, pairwise_distances)
        )

    def fit(self, epochs=10, learning_rate=1e-3, quiet=False):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.9)
        )

        for epoch in range(epochs):
            self.train()
            running_loss = 0
            start = time.time()
            # for i in range(epochs):

            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            optimizer.step()
            running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy() / 1), 5)
            if not quiet:
                print(
                    f"Epoch {epoch+1} average loss: {average_loss}"
                    + f" ({round(end-start,2)} seconds)"
                )

        return


class vanc_density:
    def __init__(
        self, tbw_m, tbw_sd, bmi_m, bmi_sd, age_m, age_sd, sex_m, scr_m, scr_sd
    ):

        self.tbw_m = tbw_m
        self.tbw_sd = tbw_sd
        self.bmi_m = bmi_m
        self.bmi_sd = bmi_sd
        self.age_m = age_m
        self.age_sd = age_sd
        self.sex_m = sex_m
        self.scr_m = scr_m
        self.scr_sd = scr_sd

        self.density = multivariate_normal(
            mean=[tbw_m, bmi_m, age_m, scr_m], cov=[tbw_sd, bmi_sd, age_sd, scr_sd]
        )

    def pdf(self, X):

        return self.density.pdf(X)

    def sample(self, N):

        return self.density.rvs(size=N)


class SMC_Vancomycin:
    def __init__(self, covariates, pred_matrix):

        self.model_predictions = pred_matrix
        self.covariates = covariates

        self.adane = vanc_density(147.9, 13.1, 49.5, 5.2, 43.0, 7.5, 0.61, 124.8, 14.0)
        self.mangin = vanc_density(82.0, 21.0, 28.0, 7.0, 63.0, 23.0, 0.87, 130.0, 50.0)
        self.medellin = vanc_density(
            72.0, 15.0, 27.5, 5.0, 74.3, 14.0, 0.45, 90.5, 52.0
        )
        self.revilla = vanc_density(73.0, 13.3, 26.2, 4.1, 61.1, 16.3, 0.66, 86.1, 55.1)
        self.roberts = vanc_density(74.8, 15.8, 25.9, 5.4, 58.1, 14.8, 0.62, 90.7, 60.4)
        self.thomson = vanc_density(72.0, 30.0, 25.9, 5.4, 66.0, 20.0, 0.63, 98.0, 51.0)

        self.list_densities = [
            self.adane,
            self.mangin,
            self.medellin,
            self.revilla,
            self.roberts,
            self.thomson,
        ]

        self.rep_learner = VancRepLearner(
            self.covariates, self.model_predictions, self.list_densities
        )

        self.rep_flag = False

    def fit_representation(self, epochs=10, learning_rate=1e-3, quiet=False):
        self.rep_learner.fit(epochs, learning_rate, quiet)

    def remodel_in_z(self, n_samples=1000):
        self.kdes = []
        for density in self.list_densities:
            samples = torch.tensor(density.sample(n_samples))
            _, latents, _ = self.rep_learner.forward(samples.float())
            latents = latents.detach().numpy()

            kernel = gaussian_kde(latents.T, bw_method="silverman")

            self.kdes.append(kernel)

        self.rep_flag = True

        return

    def get_weights_in_z(self, X, epsilon=1e-10):
        N = X.shape[0]
        weights = np.zeros((N, 6))

        _, test_latents, _ = self.rep_learner.forward(torch.tensor(X[:, 0, :4]).float())

        for i, kde in enumerate(self.kdes):
            density = kde(test_latents.detach().T)
            weights[:, i] = density

        weights += epsilon

        return weights / weights.sum(axis=1).reshape((N, 1))

    def get_weights_no_rep(self, X, epsilon=1e-10):

        N = X.shape[0]
        weights = np.zeros((N, 6))

        weights[:, 0] = self.adane.pdf(X[:, 0, [True, True, True, True, False]])
        weights[:, 1] = self.mangin.pdf(X[:, 0, [True, True, True, True, False]])
        weights[:, 2] = self.medellin.pdf(X[:, 0, [True, True, True, True, False]])
        weights[:, 3] = self.revilla.pdf(X[:, 0, [True, True, True, True, False]])
        weights[:, 4] = self.roberts.pdf(X[:, 0, [True, True, True, True, False]])
        weights[:, 5] = self.thomson.pdf(X[:, 0, [True, True, True, True, False]])

        weights += epsilon

        return weights / weights.sum(axis=1).reshape((N, 1))

    def get_weights(self, X):
        if self.rep_flag == True:
            return self.get_weights_in_z(X)
        else:
            return self.get_weights_no_rep(X)

    def pred(self, X, epsilon=1e-10):

        weights = self.get_weights(X)
        # weights += epsilon

        predictions = (self.model_predictions * weights).sum(axis=1)

        return predictions

    def pred_combo(self, X, quality_info, epsilon=1e-10):

        smc_weights = self.get_weights(X)
        # smc_weights += epsilon

        N = X.shape[0]
        weights = quality_info[:, :, 3] / quality_info[:, :, 3].sum(axis=1).reshape(
            (N, 1)
        )

        weights = (weights + smc_weights) / 2

        predictions = (self.model_predictions * weights).sum(axis=1)

        return predictions

    def pred_basic(self, X):

        N = X.shape[0]
        weights = np.ones((N, 6))
        weights = weights / weights.sum(axis=1).reshape((N, 1))

        predictions = (self.model_predictions * weights).sum(axis=1)

        return predictions

    def pred_bma(self, X, quality_info):

        N = X.shape[0]
        weights = quality_info[:, :, 3] / quality_info[:, :, 3].sum(axis=1).reshape(
            (N, 1)
        )

        predictions = (self.model_predictions * weights).sum(axis=1)

        return predictions
