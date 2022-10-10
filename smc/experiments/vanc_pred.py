from smc.models import SMC_Vancomycin
import numpy as np
from tqdm.notebook import tqdm
from pkg_resources import resource_filename
import torch
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--no-subsample", dest="subsample", action="store_false")
args = parser.parse_args()

np.random.seed(41310)
torch.manual_seed(41310)

subsample = args.subsample
if subsample:
    num_points = 1000
else:
    num_points = 6000

RESULTS_LOC = resource_filename("smc", "results/vanc_results.csv")

DATA_LOC = resource_filename("smc", "data_loading")

preds = np.load(
    (DATA_LOC + "/Vancomycin/vancomycin_processed_new.npy"), allow_pickle=True
)

scenes = ["A priori", "One Occasion (close)", "Two Occasions (1+2)", "1+2+3"]
scenes = [
    "A priori",
    "One Occasion (far)",
    "One Occasion (close)",
    "3",
    "Two Occasions (1+2)",
    "Two Occasions (2+3)",
    "1+2+3",
    "1+2+3+4",
]
scenes = [
    "A priori",
    "One Occasion (close)",
    "Two Occasions (1+2)",
    "1+2+3",
    "1+2+3+4",
]

(pred_matrix, auc_true, covariates, model_n, quality_info) = preds.item()["A priori"]

smc = SMC_Vancomycin(covariates, pred_matrix)
best_model = smc.get_weights(covariates).argmax(axis=1)
if subsample:
    mask = (best_model + 1) == model_n
else:
    mask = np.full(6000, True)

smc.fit_representation(epochs=10000, learning_rate=1e-2)
smc.remodel_in_z(n_samples=1000)


rss_dict = {}
for scene in tqdm(scenes):
    print(scene)
    (pred_matrix, auc_true, covariates, model_n, quality_info) = preds.item()[scene]

    new_pred_matrix = np.zeros((6000, 10))
    new_pred_matrix[:, :6] = pred_matrix

    smc.model_predictions = pred_matrix

    smc.rep_flag = False

    my_preds = smc.pred(covariates)
    basic_preds = smc.pred_basic(covariates)
    bma_preds = smc.pred_bma(covariates, quality_info)
    combo_preds = smc.pred_combo(covariates, quality_info)
    new_pred_matrix[:, 6] = basic_preds
    new_pred_matrix[:, 7] = bma_preds
    new_pred_matrix[:, 8] = my_preds
    new_pred_matrix[:, 9] = combo_preds

    masked_pred = new_pred_matrix[mask, :]
    masked_auc = auc_true[mask]

    num = 10
    rss_matrix = np.zeros((10, num))
    for i in range(num):
        indx = slice(i, num_points, num)
        mask2 = np.full(num_points, True)
        mask2[indx] = False
        indx = mask2
        rss = np.sqrt(
            (
                ((masked_pred[indx] - masked_auc[indx].reshape(-1, 1)) ** 2)
                / (masked_auc[indx].reshape(-1, 1) ** 2)
            ).mean(axis=0)
        )
        rss_matrix[:, i] = rss
    rss_dict[scene] = rss_matrix

header = [
    "",
    "Adane",
    "",
    "Mangin",
    "",
    "Medellin",
    "",
    "Revilla",
    "",
    "Roberts",
    "",
    "Thomson",
    "",
    "Naive",
    "",
    "PBMA",
    "",
    "SMC",
    "",
    "SMC+PBMA",
]

with open(RESULTS_LOC, "w") as f:
    writer = csv.writer(f)
    writer.writerow(header[:15] + header[17:19])

    for scene in tqdm(scenes):
        print(scene)
        rss_matrix = rss_dict[scene]
        means = rss_matrix.mean(axis=1)
        stds = rss_matrix.std(axis=1)
        result = [scene]
        for i in range(10):
            print(f"$${(means[i]*100).round(1)} \pm {(stds[i]*100).round(1)}$$")
            result.append((means[i] * 100).round(1))
            result.append((stds[i] * 100).round(1))
        writer.writerow(result[:15] + result[17:19])
