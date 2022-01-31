This repository contains the experiments for "Hybrid Semiparametric Bayesian Networks."

There is a folder for each experiment type. `synthetic` for synthetic data experiments, and `UCI data` for experiments from the UCI repository.

To run these experiments [`PyBNesian`](https://github.com/davenza/PyBNesian) is needed. The experiments were run on a modified v0.4.0 version. The patch applied to PyBnesian v0.4.0 is in `pybnesian_patch/hspbn_experiments.patch`. This patch includes the implementation of BDeu for discrete factors. Also, the patch controls the existence of discrete configurations that are in the test data, but not in the training data. We also provide compiled wheels in the folder `pybnesian_patch` for several Python and operatoring system versions.

The experiments were run on Ubuntu 16.04, Python 3.7.3, and compiled with gcc 9.3.0. We believe that the results should be reproducible for most configurations. Altough sometimes C++ compilers introduce optimizations and changes in the standard library implementation in new releases. These changes may subtly affect the results.

Organization
=================

Synthetic Experiments
---------------------

The synthetic experiments contain the following files:

- `util.py`
- `generate_new_bns.py`
- `generate_dataset.py`
- `train_hc_[model_type].py`
- `test_hc_[model_type].py`
- `test_hc_times.py`

`util.py` defines the parameters of the experiment at the start. Also, it contains some auxiliary code used for the experiments.

`generate_new_bns.py` generates random Bayesian networks and includes auxiliary classes and methods. This is the first script that should be called. It saves the models in a local folder called `ground_truth_models/` (ensure the folder exists).

`generate_dataset.py` generates all training and test datasets from the random Bayesian networks. You should execute `python generate_dataset_hspbn.py` after `python generate_new_bns.py`. It saves the datasets in a local folder called `data/` (ensure the folder exists).

The experiments are executed in two steps so the experiments can be paused/resumed easily. First, the models are learned from the training data and saved. For this, use the scripts `train_hc_[model_type].py`. This step can take quite some time, so the execution can be stopped at any moment for all training scripts. If the script is executed again, it will automatically detect the already learned models. All the learned models are saved in a local folder called `models/`. All the learned models (including all the iterations of the greedy hill-climbing algorithm) are saved.

The `[model_type]` can take the following values:

- `clg`: to learn conditional linear Gaussian Bayesian networks.
- `hspbn`: to learn HSPBNs with CLG CPDs at the start.
- `hspbn_hckde`: to learn HSPBNs with HCKDE CPDs at the start.
 
Then, the `test_hc_[model_type].py` scripts load the learned models and test them on unseen data. The results of the experiments are printed on the screen.

The `test_hc_times.py` script summarizes the average learning runtime for each model type. The training time of each model is measured by the `train_hc_[model_type].py` scripts and saved in the corresponding `model/` folder. Ensure you train the models with `PARALLEL_THREADS = 1` in `util.py` to obtain representative results.

UCI Data
--------

The UCI data experiments contain the following files:

- `util.py`
- A Python file for each dataset tested.
- `plot_results.py`
- `adjusted_pvalues.py`
- `plot_cd_diagram.py`

`util.py` defines the parameters of the experiment at the start. Also, it contains some auxiliary code used for the experiments.

Each dataset also has a corresponding Python file. Calling one of this files, trains all the models for this dataset. As in the synthetic experiments, it saves all the models in the local folder `models/`. Then, it evaluates the performance of all models on unseen data and prints the results on the screen.

`plot_results.py` saves a `result_summary.csv` file which contains the results for each dataset and algorithm. Then, it plots the CD diagram comparing all the algorithms in a local folder called `plots/`. **You can call this file after training all the models for all the datasets**. That is, you must execute all the dataset scripts before calling `plot_results.py`

`adjusted_pvalues.py` and `plot_cd_diagram.py` is auxiliary code for `plot_results.py`. They perform multiple hypothesis tests and plot CD diagrams.
