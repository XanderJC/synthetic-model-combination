import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import resource_filename

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 14

RESULTS_LOC = resource_filename("smc", "results/")

aucs = np.load(RESULTS_LOC + "MNIST_pred_results_CR.npy", allow_pickle=True)

means = aucs.mean(axis=1)
stds = aucs.std(axis=1)

n_samples = [1, 3, 5, 7, 10, 20, 50, 100, 500, 1000, 5000]

means = np.array([0.5] + list(means))
stds = np.array([0.1] + list(stds))

print(means)
print(stds)


CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"

fig, ax = plt.subplots()

ax.plot(n_samples, means, color=CB91_Amber, linestyle="--", label="SMC")
ax.scatter(n_samples, means, color=CB91_Amber, marker="x")  # type: ignore
ax.fill_between(
    n_samples,
    means - 2 * stds,  # type: ignore
    means + 2 * stds,  # type: ignore
    alpha=0.2,
    color=CB91_Amber,
)

ax.set_xscale("log")
ax.set_ylim([0.45, 1])  # type: ignore
ax.set_xlim([1, 5000])  # type: ignore
ax.set_xticks([10, 100, 1000])
ax.set_xlabel("Num. of Samples (log scale)")
ax.set_ylabel("AUROC")
ax.set_title("AUROC as Information Increases")

ax.axhspan(0.51 - 0.01, 0.51 + 0.01, alpha=0.2, color=CB91_Purple)
ax.axhline(0.51, color=CB91_Purple, linestyle="--", label="Uncertainty Weighting")

ax.axhspan(0.5 - 0.01, 0.5 + 0.01, alpha=0.2, color=CB91_Green)
ax.axhline(0.5, color=CB91_Green, linestyle="--", label="Majority Voting")

ax.legend(loc="center right")

plt.grid(ls="--", which="both", alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_LOC + "MNIST_pred.pdf")
plt.show()
