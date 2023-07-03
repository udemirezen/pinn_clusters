# pinn_clusters
Accompanying code and results for paper: "1D Ice Shelf Hardness Inversion: Clustering Behavior and Collocation Resampling in
Physics-Informed Neural Networks" by Yunona Iwasaki and Ching-Yao Lai. Code for training PINNs for 1D ice-shelf inverse modeling and analysis of training results over repeated trials.

# Installation

# Table of Contents
## pinn_trial.py
Main script for training PINNs to predict for the correct 1D $u(x)$ (velocity), $h(x)$ (thickness), and $B(x)$ (hardness) profiles given synthetic noisy data for $u(x)$ and $h(x)$. In addition to training PINNs, this script handles the generation of synthetic noisy training data at a specified noise level, as well as evaluating PINN predictive accuracy compared to ground truth profiles. 

This script requires the user to specify the ground truth $u(x)$, $h(x)$, and $B(x)$ profiles. Currently, $u(x)$ and $h(x)$ profiles are specified by providing a reference to a Python dictionary saved as a ```.mat``` file using the ```N_t``` variable (line 35). The ground truth $B(x)$ profile is specified by passing an array of values to the ```B_truth``` variable  (line 39).

Additionally, this script allows users to specify the following hyperparameters relevant to the study:

* N_t _(int)_: Number of collocation points. This number stays fixed, even if the script switches between collocation resampling and fixed collocation points (line 43)

* layers _(list)_: List specifying the width and depth of the neural network. Specify the size of each layer except for the input layer. e.g. ```layers = [5,5,3]``` for a neural network with two, 5-unit hidden layers. The final value specifies the size of the output layer and must be set to 3 for this problem. (line 47)

* num_iterations_adam_resample, num_iterations_adam_fixed, num_iterations_lbfgs _(int)_: Specify the number of iterations to train with each optimizer and collocation method. (lines 53-55)
  * ```adam_resample```: train with Adam optimizer using collocation resampling.
  * ```adam_fixed```: train with Adam optimizer wih fixed collocation points
  * ```lbfgs```: train with L-BFGS optimizer with fixed collocation points.

* test_noise _(float)_: level of noise added to ground truth $u(x)$ and $h(x)$ profiles during synthetic data generation. Please refer to p. 6 of the main text for the definition of noise level; it may also be helpful to see its implementation in the script ```noise.py```.
  
_Note: there is no option to run L-BFGS with collocation resampling, as LBFGS is a second-order optimization algorithm (i.e. the update to the neural network weights is determined by the two preceding iterations); training will quickly terminate if this is attempted._

* test_gammas _(list)_: specify one or multiple values of $\gamma$ to test. Curently, to conveniently implement logarithmic spacing of $\frac{\gamma}{1-\gamma}$, we first specify $\log_{10}(\frac{\gamma}{1-\gamma})$ values using the ```logratios``` variable, then solve for the corresponding $\gamma$-values.

Additional information can be found in the line-by-line explanations provided in the code comments.


## formulations, optimization.py, loss.py, loss_colo.py, model.py
Required files with functions for training PINNs. "loss.py" and "loss_colo.py" define the loss functions used for testing. "loss.py" should be used for training with fixed collocation points, while "loss_colo.py" should be used for training using collocation resampling.

## constantB_uh.mat, sinusoidalB_uh.mat
Ground truth profiles for $u(x)$ and $h(x)$ from which noisy data is generated. "constantB_uh.mat" are the $u(x)$ and $h(x)$ solutions for $B(x) = 1.0$, $x \in [0.0,1.0]$. "sinusoidalB_uh.mat" are the numerical $u(x)$ and $h(x)$ for $B(x) = \\frac{1}{2} \cos{(3\pi x)}$, $x \in [0.0,1.0]$. Both assume boundary conditions $u(0) = 1$, $h(0) = h_0$. See p. 5 of the main text and pp. 2-3 of the supplementary material for the definition and numerical value of $h_0$ and other relevant constants. 

## trial_processing.ipynb
Jupyter notebook for consolidating error data from a set of trial result dictionaries into a single numpy array.

## pinn_cluster_plots.ipynb
Jupyter notebook that loads the numpy error array of a set of training trials and separates trials by $k$-means clustering in log-space. Code for vizualising clusters, plotting cluster statistics, etc.

## trial_results
Numpy arrays of the $B_{\mathrm{err}}$, $u_{\mathrm{err}}$, $h_{\mathrm{err}}$ for different experiments studied in the paper. Each numpy array has shape $(n, m, l)$, where $n$ is the number of values of $\gamma$ tested in the experiment, $m = 3$ is the number of predictive variables (i.e. $u$, $h$, $B$), and $l$ is equal to the number of repeated trials. In this repo, $l=501$ for all experiments. Please use the following code to load each array:

```
errors = np.load('path_to_file/errors.npy')

gi = 1

u_errs = errors[gi][0]
h_errs = errors[gi][1]
B_errs = errors[gi][2]
```
where ```gi``` should be modified to the index of the $\gamma$-value being examined.

Naming conventions for each numpy array are as follows:

### clean_u206l1kc_errs.npy, nx_u206l1kc_errs.npy
Error results using the standard settings(six, 20-unit hidden layers with $c=1001$ fixed collocation points). The prefix "clean" corresponds to tests using clean training data; prefixes "nx" denote tests using noisy training data, with x specifying the level of noise (i.e. n3_u206l1kc_errs.npy corresponds to noise = 0.3; n05_u206l1kc_errs.npy corresponds to noise = 0.05)

### resampled_fixed.npy
Results from training with collocation resampling followed by fixed collocation points for noise level $= 0.3$, (see Section 3.4 of the main text, pp. 14-15).
We tested $l = 12$ values of $\gamma$ such that $\frac{\gamma}{1-\gamma}$ are logarithmically spaced between $[10^{-4}, 10^6]$, i.e., ```gi = 0``` corresponds to $\frac{\gamma}{1-\gamma} = 10^{-4}$, ```gi = 1``` corresponds to $\frac{\gamma}{1-\gamma} = 10^{-3}$, ...```gi = 11``` corresponds to $\frac{\gamma}{1-\gamma} = 10^{7}$. Note that we omit the largest value of $\gamma$ tested in the experiments using fixed collocation points, so that we have one fewer value of $\gamma$. 


### ux_errs
Results from tests with increased neural network width. ux corresponds to x-units per hidden layer. (Noise level = 0.3 for both)
### u100_c1k_errs.npy
Results from final test with two 100-hidden units and 1001 collocation points. (Noise level = 0.3)

