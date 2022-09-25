# pinn_clusters
Accompanying code for paper: "Clustering Behaviour of Physics-Informed Neural Networks: Inverse Modeling of An Idealized Ice Shelf." Code for training PINNs for 1D ice-shelf inverse modeling and analysis of training results over repeated trials.

## pinn_trial.py
Code for training PINNs.

## /formulations, optimization.py, loss.py, model.py
Required files with functions for training PINNs.

## truth_uh.mat 
Ground truth profiles for u and h from which noisy data is generated.

## trial_processing.ipynb
Jupyter notebook for consolidating error data from a set of trial result dictionaries into single Numpy array.

## pinn_cluster_plots.ipynb
Jupyter notebook that loads the Numpy error array of a set of training trials and separates trials by k-means clustering in log-space. Code for vizualising clusters, plotting cluster statistics, etc.

## trial_results
Numpy arrays of the errors separated by noise level. Each noise level was tested for 13 gamma values.
### anc_errs, anx_errs
Error results using the standard settings. anc corresponds to clean data, anx corresponds to noise level = 0.x
### ux_errs
Results from tests with increased neural network width. ux corresponds to x-units per hidden layer. (Noise level = 0.3 for both)
### u100_c1k_errs.npy
Results from final test with two 100-hidden units and 1001 collocation points. (Noise level = 0.3)

