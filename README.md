# [Synthetic Model Combination: An Instance-wise Approach to Unsupervised Ensemble Learning](https://openreview.net/forum?id=RgWjps_LdkJ)

### Alex J. Chan and Mihaela van der Schaar

### Advances in Neural Information Processing Systems (NeurIPS) 2022

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Last Updated: 10 Oct. 2022

Code Author: Alex J. Chan (ajc340@cam.ac.uk)

This repo is pip installable - clone it, optionally create a virtual env, and install it:

```shell
git clone https://github.com/XanderJC/synthetic-model-combination.git

cd synthetic-model-combination

pip install -e .
```

## Reproducing experimental demonstration

All RNG seeds are set in scripts and pretrained models provided so they should produce exact results.

Requirements to run experiments can be found in 'requirements/requirements.txt', using Python 3.8.8.

### Regression Example

Running the following Jupyter notebooks go through the synthetic regression example from scratch:

- smc/experiments/example.ipynb
- smc/experiments/example_uncertainty.ipynb
- smc/experiments/example_high_overlap.ipynb


### MNIST

To produce Figure 5, run:

```shell
python smc/experiments/MNIST_pred_results.py
```
Which uses results generated from:

- smc/experiments/mnist_prediction.py

This script loads results and pretrained models for both the ensemble members and the SMC representation which can themselves be trained with the following scripts respectively:

- smc/experiments/mnist_model_training.py
- smc/experiments/mnist_rep_learn.py

### Vancomycin

To produce Table 2, run:

```shell
python smc/experiments/vanc_pred.py
```

Which will print results to the console but also save to 'vanc_results.csv'.

### Citing 

If you use this software please cite as follows:

```
@inproceedings{chan2022synthetic,
    title={Synthetic Model Combination: An Instance-wise Approach to Unsupervised Ensemble Learning},
    author={Alex James Chan and Mihaela van der Schaar},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=RgWjps_LdkJ}
}
```
