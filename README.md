# pinn_clusters
Accompanying code and results for the paper: "1D Ice Shelf Hardness Inversion: Clustering Behavior and Collocation Resampling in
Physics-Informed Neural Networks" [link to paper] by Yunona Iwasaki and Ching-Yao Lai. We document the codes used to train PINNs for 1D ice-shelf inverse modeling. Additionally,  we provide scripts to facilitate analysis of training results over repeated trials.

In our study, we investigated the feasbility of a physics-informed neural network to invert for ice-shelf hardness spatial profile $B(x)$ given sparse, noisy training data of only velocity $u(x)$  and thickness $h(x)$. We introduced a weighting hyperparameter $\gamma$ in the objective function of the PINN, adjusting the relative weighting between equation and data loss.  
![Highlight clustering for default training scheme](https://github.com/YaoGroup/pinn_clusters/blob/main/clusterplot_forgithub.png)
*Figure 1. The clustering behaviour of  PINN predictions. Correlation of $B_{err}$ with $u_{err}$ and $h_{err}$ over 501 trials for various values of $\frac{\gamma}{1-\gamma}$ and noise level = 0.3, using six, 20-unit hidden layers and $c=1001$ collocation points. PINNs were trained for 400,000 iterations of Adam and 200,000 iterations of LBFGS, or until convergence. (a),(b): $u_{err}$ and $h_{err}$ vs. $B_{err}$, respectively.*
Meanwhile,
![Highlight clustering for random collocation resampling](https://github.com/YaoGroup/pinn_clusters/blob/main/collocation_resampling_highlight.png)

Please direct questions about this code to Yunona Iwasaki: yiwasaki [at] berkeley [dot] edu .


# Installation

# Table of Contents
## pinn_trial.py
Main script for training PINNs to predict for the correct 1D $u(x)$ (velocity), $h(x)$ (thickness), and $B(x)$ (hardness) profiles given synthetic noisy data for $u(x)$ and $h(x)$. In addition to training PINNs, this script handles generation of synthetic noisy training data at a specified noise level, as well as evaluation of PINN predictive accuracy compared to ground truth profiles. 

This script requires the user to specify the ground truth $u(x)$, $h(x)$, and $B(x)$ profiles. Currently, $u(x)$ and $h(x)$ profiles are specified by providing a reference to a Python dictionary saved as a ```.mat``` file using the ```N_t``` variable (line 35). The ground truth $B(x)$ profile is specified by passing an array of values to the ```B_truth``` variable  (line 39). ```B_truth``` should contain the ground truth values of $B(x)$ at the same values of $x$ as the ground truth $u(x)$ and $h(x)$ profiles.

Additionally, this script requires users to specify the following hyperparameters relevant to our optimization study:

* N_t _(int)_: Number of collocation points. This number stays fixed, even if the script switches between collocation resampling and fixed collocation points (line 43).

* layers _(list)_: List specifying the width and depth of the neural network. Specify the size of each layer except for the input layer. e.g. ```layers = [5,5,3]``` for a neural network with two, 5-unit hidden layers. The final value specifies the size of the output layer and must be set to 3 for this problem. (line 47)

* num_iterations_adam_resample, num_iterations_adam_fixed, num_iterations_lbfgs _(int)_: Specify the number of iterations to train with each optimizer and collocation method (lines 53-55).
  * ```adam_resample```: train with Adam optimizer using collocation resampling.
  * ```adam_fixed```: train with Adam optimizer wih fixed collocation points
  * ```lbfgs```: train with L-BFGS optimizer with fixed collocation points.

* test_noise _(float)_: level of noise added to ground truth $u(x)$ and $h(x)$ profiles during synthetic data generation. Please refer to p. 6 of the main text for the definition of noise level; it may also be helpful to see its implementation in the script ```noise.py```.
  
_Note: We have not included the option to run L-BFGS with collocation resampling, as L-BFGS is a second-order optimization algorithm (i.e. the update to the neural network weights is determined by the __two__ preceding iterations); training will quickly terminate if this is attempted._

* test_gammas _(list)_: specify one or multiple values of $\gamma$ to test. To conveniently implement logarithmic spacing of $\frac{\gamma}{1-\gamma}$, the user may the specify $\log_{10}(\frac{\gamma}{1-\gamma})$ values using the ```logratios``` variable, then solve for the corresponding $\gamma$-values in the next line (lines 221-222).

Additional information can be found in the line-by-line explanations provided in the code comments.

## loss.py

Contains the loss functions used in the paper, namely ```SquareLoss``` used for fixed collocation points, and ```SquareLossRandom``` used for collocation resampling. Please refer to the final section of this README (__"Code Implementation of Collocation Resampling"__) for a detailed explanation of how these two functions differ.

Both loss functions evaluate the predictive accuracy of the neural network after each iteration according to the characteristic objective function of PINN, which we call $J(\Theta)$: ([Raissi et. al, 2019](https://doi.org/10.1016/j.jcp.2018.10.045))

<p align="center">
$J(\Theta) = \gamma E(\Theta) + (1-\gamma)D(\Theta)$
</p>

where we introduce an additional hyperparameter $\gamma \in [0.0, 1.0]$ to adjust the relative weighting between the equation loss $E(\Theta)$ and the data loss $D(\Theta)$. $D(\Theta)$ is evaluated at values in the domain where training data is available, while $E(\Theta)$ is evaluated at a set of collocation points sampled _independently_ of the available training data. Please see p. 3 of the main text for the precise definitions of equation and data loss used in our paper; p. 5 of the main text for the governing physics equations enforced by $E(\Theta)$, and ```formulations/eqns_o1_inverse.py``` for the implementation of these equations in our codes.

### Initialization
An instance of the ```SquareLoss``` function is initialized by the following code:
```
loss = SquareLoss(equations=physics_equations, equations_data=data_equations, gamma=gamma)
```
where
* equations: An iterable of callables with the signature ```function(x, neuralnet)``` corresponding to the governing physics equations. To enforce 1D SSA, we pass ```Inverse_1stOrder_Equations``` imported from ```formulations/eqns_o1_inverse.py``` .
*  equations_data: An iterable of callables with the signature ```function(x, neuralnet)``` corresponding to the governing physics equations. We use ```Data_Equations``` imported from ```formulations/eqns_o1_inverse.py```.
*  gamma (_float_): the value of $\gamma$ with which to evaluate the objective function $J(\Theta)$.

```SquareLossRandom``` is initialized with the same arguments.
## formulations

## optimization.py

## model.py


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

# Code Implementation of Collocation Resampling

Training can be switched between using fixed collocation points and collocation resampling by switching the loss function used during training. The loss function evaluated by a given optimizer is specified during the initialization of the optimizer. Use the  ```SquareLoss``` loss function when using fixed collocation points, and ```SquareLossRandom``` for random collocation resampling (see lines 77-100 in 'pinn_trials.py').

Comparing the ```SquareLoss``` and ```SquareLossRandom``` functions in 'loss.py', the main difference between the two functions is in the ```__call__``` method. For ```SquareLossRandom```, we add a few extra lines at the beginning of the  ```__call__``` method (lines 54-61):

```
def __call__(self, x_eqn, data_pts, net) -> Dict[str, tf.Tensor]:
    xmin = 0.0
    xmax = 1.0
    N_t = 1001
    _data_type = tf.float64       
    collocation_pts = xmin + (xmax - xmin) * self.col_gen.uniform(shape = [N_t])
    collocation_pts = collocation_pts**3
```
where ```self.col_gen``` is a stateful random generator defined in the ```__init__``` method (line 52):

```
        self.col_gen = tf.random.get_global_generator()
```
Thus, the ```SquareLossRandom``` function generates a new set of collocation points every time it is called, i.e. at every iteration. 

__Important Note: It is essential to use a _stateful_ random number generator such as ```tf.random.Generator()``` to ensure that the collocation points are resampled after each iteration.__ Using a stateless random generator (such as 
 those provided in the ```numpy.random``` module, or the ```lhs``` generator used in our codes for fixed collocation point generation) will not allow the collocation points to be updated in a TensorFlow training loop, causing the loss function to behave identically to training with fixed collocation points.

