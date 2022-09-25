import sys
import time
from pathlib import Path
import math as m

sys.path.append("/home/yiwasaki/IceShelf1D")

#indexing for parallelizing multiple trials at fixed noise + gamma range using SLURM 
import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

expnum = str(idx) #experiment number label for result files

import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt

from data.noise import add_noise
from data.sample import random_sample
from model import create_mlp
from loss import SquareLoss
from optimization import LBFGS, Adam
from formulations.constants import *
from formulations.helpers import get_collocation_points, to_mat_tensor, to_tensor
from formulations.helpers import _data_type

"""Determine which equations to use"""
from formulations.eqns_o1_inverse import Data_Equations, Inverse_1stOrder_Equations

"""Define the domain of the problem"""

#Import ground truth data for u,h and their x-positions (x) from which to build synthetic noisy training data
data = loadmat('/home/yiwasaki/IceShelf1D/reproduction_data/constant_ground_truth_on_cluster_Adam10000LBFGS75000Gamma0.01.mat')
x_star = np.transpose(data['x']) 
u_star = np.transpose(data['u'])[:, 0]
h_star = np.transpose(data['h'])[:, 0]

"""Parameters"""
# Data parameters
N_t = 201  # Number of collocation points
N_ob = 401  # Number of training points.

# Model parameters
layers = [5, 5, 3] #number of units in each layer.
lyscl = [1, 1, 1] #standard deviation to set the scales for Xavier weight initialization

# Hyper parameters for the PINN
fractional = False
num_iterations_adam = 400000
num_iterations_lbfgs = 200000

#function to train PINNs at a single value of gamma and noise level. Takes as argument ground truth values of u and h (u_star and h_star) at x locations
#given by x_star. Adds noise to u and h data before feeding to PINN as training data.
def Berr_func(gamma, noise_level, x_star, u_star, h_starcales set the standard deviation of the (Xavier) weight initialization layers):
    collocation_pts = get_collocation_points(x_train=x_star, xmin=x_star.min(), xmax=x_star.max(), N_t=N_t)
    model = create_mlp(layers, lyscl, dtype=_data_type)
    equations = Inverse_1stOrder_Equations(fractional=fractional)
    loss = SquareLoss(equations= equations, equations_data=Data_Equations, gamma=gamma)

    # add noise and randomly sample the data points
    x_sampled, u_sampled, h_sampled = random_sample(
        N_ob, x_star,
        add_noise(u_star, ratio=noise_level),
        add_noise(h_star, ratio=noise_level)
    )

    # runs the model for our choice of optimizer
    start_time = time.time()
    adam = Adam(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    adam.optimize(nIter=num_iterations_adam)
    lbfgs = LBFGS(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    lbfgs.optimize(nIter=num_iterations_lbfgs)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
    
    equation_losses = np.array(adam.loss_records.get("loss_equation", []) + lbfgs.loss_records.get("loss_equation", []))
    data_losses = np.array(adam.loss_records.get("loss_data", []) + lbfgs.loss_records.get("loss_data", []))
    total_losses = np.array(adam.loss_records.get("loss", []) + lbfgs.loss_records.get("loss", []))

    data_losses = np.trim_zeros(data_losses, 'b')
    equation_losses = np.trim_zeros(equation_losses, 'b')
    total_losses = np.trim_zeros(total_losses, 'b')

    
    ############################# Data Processing ###############################

    x_star = tf.cast(x_star, dtype=_data_type)
    uhb_pred = model(x_star)
    f_pred = equations(x_star, model, drop_mass_balance=False)
    x_star = x_star.numpy().flatten().flatten()
    x_sampled = x_sampled.flatten().flatten()
    u_star = u_star.flatten()
    u_sampled = u_sampled.flatten()
    h_star = h_star.flatten()
    h_sampled = h_sampled.flatten()
    u_p = uhb_pred[:, 0:1].numpy().flatten()
    h_p = uhb_pred[:, 1:2].numpy().flatten()
    B_p = uhb_pred[:, 2:3].numpy().flatten()
    mom_residue = f_pred[0].numpy().flatten()
    mass_residue = f_pred[1].numpy().flatten()
    B_truth = np.ones_like(x_star)
    total_berr = Berr(B_p, B_truth)

    #return a dictionary of results
    results = { "total_berr" : total_berr,
                "B_p" : B_p, 
                "data_losses" : data_losses, 
                "equation_losses" : equation_losses, 
                "total_losses" : total_losses, 
                "u_p" : u_p,
                "h_p" : h_p,
                "u_sampled" : u_sampled,
                "h_sampled" : h_sampled
               }   
    return results
    N = B_p.size
    return (1/N)*np.sum(np.square(B_p-B_truth))

def gamma_batch(test_gammas, noise_level, x_star, u_star, h_star, layers): 
    batch_results = []
    for i, gamma in enumerate(test_gammas):
        #store result of training into dictionary
        exp_dic = Berr_func(gamma, noise_level, x_star, u_star, h_star, layers)
        batch_results.append(exp_dic)
    return batch_results

#select gammas to test: space gamma values logarithmically from 10^-4 to 10^8
logratios = np.linspace(-4,8,13)

#solve for gamma --> r = x/1-x ==> r-rx = x ==> r = x+rx ==> r = x(1+r) ==> x = r/(1+r)
test_gammas = np.power(10,logratios)/(1+np.power(10,logratios))

def format_dict(dict_list):
    berrs = []
    bpreds = []
    d_losses = []
    e_losses = []
    t_losses = []
    u_preds = []
    h_preds = []
    u_samp = []
    h_samp = []

    for i in range(len(dict_list)):
        berrs.append(dict_list[i]["total_berr"])
        bpreds.append(dict_list[i]["B_p"])
        d_losses.append(dict_list[i]["data_losses"])
        e_losses.append(dict_list[i]["equation_losses"])
        t_losses.append(dict_list[i]["total_losses"])
        u_preds.append(dict_list[i]["u_p"])
        h_preds.append(dict_list[i]["h_p"])
        u_samp.append(dict_list[i]["u_sampled"])
        h_samp.append(dict_list[i]["h_sampled"])

    new_dict = {"berrs" : np.asarray(berrs),
                "bpreds" : np.asarray(bpreds),
                "d_losses" : d_losses, #just keep as list
                "e_losses" : e_losses, #just keep as list
                "t_losses" : t_losses, #just keep as list
                "u_p" : np.asarray(u_preds),
                "h_p" : np.asarray(h_preds),
                "u_sampled" : np.asarray(u_samp),
                "h_sampled" : np.asarray(h_samp)
               }
    return new_dict
test_noise = 0.3
results = gamma_batch(test_gammas, test_noise, x_star, u_star, h_star, layers) #test a range of gammas for noise = 0.3
result_dict    = format_dict(results)

from scipy.io import savemat
file_str = "r" + expnum #label results by the experiment trial number
savemat('/home/pinntrial_results/' + file_str + '.mat', result_dict) #save trial results to folder **MODIFY TO DESIRED RESULTS DIRECTORY**

