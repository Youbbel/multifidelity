from __future__ import annotations


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

import trieste
import tensorflow as tf
import numpy as np
from trieste.acquisition import (
    AcquisitionFunction,
    SingleModelAcquisitionBuilder,
    Product,
)
from typing import Optional, cast, Generic, TypeVar, Sequence
from trieste.data import Dataset
from trieste.types import TensorType
import tensorflow_probability as tfp
from trieste.acquisition import AcquisitionFunctionClass
#from trieste.objectives import scaled_branin
from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import (
    ProbabilisticModel,
    TrainableProbabilisticModel,
)
# from trieste.acquisition.optimizer import generate_continuous_optimizer
from gpflow.logdensities import multivariate_normal
import gpflow
from gpflow.models import GPR
import math
import matplotlib.pyplot as plt

from trieste.data import add_fidelity_column
from trieste.data import split_dataset_by_fidelity


os.chdir('/user/abellouc/home/Post-Doc/multi_fidelity/simulation/tree_fidelities/new_simulations/3level_fidelities_ratiomesh_1_3_10')

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 12})

filename = "data.pickle"
file_time='observation.txt'

DOE_bf=40
DOE_mf=25
DOE_hf=10
EGO_phase=250

n_iter=DOE_bf+DOE_hf+DOE_mf+EGO_phase

with open(filename, 'rb') as file:
    dataset = pickle.load(file)

with open(file_time, 'r') as file:
    content = file.read()

import re
wall_times = re.findall(r'Wall time\s*:\s*([\d.]+)', content)

wall_times = [float(time) for time in wall_times]
wall_times = wall_times[-(DOE_bf+DOE_hf+DOE_mf+EGO_phase)*2:]

total_time_seconds = sum(wall_times)

time_iter=np.zeros(n_iter)
cum_time=np.zeros(n_iter)
for i in range(0,2*n_iter,2):
    time_iter[i//2]=wall_times[i]+wall_times[i+1]
    cum_time[i//2]=np.sum(time_iter[:i//2+1])

cum_time=cum_time/3600
time_iter=time_iter/3600
total_time=total_time_seconds/3600

def filter_by_fidelity(query_points: TensorType):

    input_points = query_points[:, :-1]  # [..., D+1]
    fidelities = query_points[:, -1:]  # [..., 1]

    lowfi_mask = (fidelities[:, 0] == 0.)
    ind_lowfi = tf.where(lowfi_mask)[:, 0]

    medfi_mask = (fidelities[:, 0] == 1.)
    ind_medfi = tf.where(medfi_mask)[:, 0]

    highfi_mask = (fidelities[:, 0] == 2.)
    ind_highfi = tf.where(highfi_mask)[:, 0]

    lowfi_points = tf.gather(input_points, ind_lowfi, axis=0)
    medfi_points = tf.gather(input_points, ind_medfi, axis=0)
    highfi_points = tf.gather(input_points, ind_highfi, axis=0)
    return lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi



lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi = filter_by_fidelity(dataset.query_points)

obj_best=tf.gather(dataset.observations, ind_highfi)[DOE_hf:]

minvalue=np.zeros(len(obj_best))
for i in range(len(obj_best)):
    minvalue[i]=min(obj_best[:i+1])

# Plot the minvalue until the end
last_ind = tf.constant([DOE_bf+DOE_mf+DOE_hf+EGO_phase-1], dtype=tf.int64)
minvalue_ind = tf.concat([ind_highfi[DOE_hf:], last_ind], axis=0)
minvalue= tf.concat([minvalue, min(obj_best[:])], axis=0)
print('effiency :',100*(1-minvalue[-1].numpy()),'%')

plt.figure(1)
plt.xlabel(r'Computation time [h]')
plt.ylabel(r'Objective function')

plt.plot(cum_time[ind_lowfi[:DOE_bf]],tf.gather(dataset.observations, ind_lowfi)[:DOE_bf],'x',color='blue', markersize=4)#,label='DOE LF')
plt.plot(cum_time[ind_medfi[:DOE_mf]],tf.gather(dataset.observations, ind_medfi)[:DOE_mf],'x',color='darkgreen', markersize=4)#,label='DOE LF')
plt.plot(cum_time[ind_highfi[:DOE_hf]],tf.gather(dataset.observations, ind_highfi)[:DOE_hf],'x',color='red', markersize=4)#,label='DOE HF')

plt.plot(cum_time[ind_lowfi[DOE_bf:]],tf.gather(dataset.observations, ind_lowfi)[DOE_bf:],'o',color='blue', label='LF', markersize=6)#,markeredgecolor='black')
plt.plot(cum_time[ind_medfi[DOE_mf:]],tf.gather(dataset.observations, ind_medfi)[DOE_mf:],'o',color='darkgreen', label='MF', markersize=6)#,markeredgecolor='black')
plt.plot(cum_time[ind_highfi[DOE_hf:]],tf.gather(dataset.observations, ind_highfi)[DOE_hf:],'o',color='red', label='HF', markersize=6)#,markeredgecolor='black')
#plt.plot(minvalue_ind[:],minvalue[:],'-',color='red',label='min value MF')


print('Nombre de simulation LF:', len(ind_lowfi[DOE_bf:]))
print('Nombre de simulation MF:', len(ind_medfi[DOE_mf:]))
print('Nombre de simulation HF:', len(ind_highfi[DOE_hf:]))

plt.plot(cum_time[minvalue_ind[:]],minvalue[:],'-',color='red',label='min value HF')


#plt.fill_between([0,cum_time[DOE_bf]], [1.005,1.005], color='blue', alpha=0.1)
plt.fill_between([0,cum_time[DOE_bf+DOE_mf+DOE_hf]], [1.005,1.005], color='violet', alpha=0.3)

import matplotlib.patheffects as path_effects
text = plt.text(cum_time[DOE_bf+DOE_mf+DOE_hf+EGO_phase-70], minvalue[-1]-0.11, r'min J$_{HF}=$ '+str(np.round(minvalue[-1],3)), fontsize = 14, color='red')

text.set_path_effects([path_effects.Stroke(linewidth=0.05, foreground='black'),
                       path_effects.Normal()])

plt.text(cum_time[DOE_bf+DOE_mf+DOE_hf-69], 0.005, r'DOE MF', fontsize = 14, color='black', alpha=0.5)

plt.xlim(0,20)
plt.ylim(0,1)
plt.tight_layout()
plt.legend(loc='upper right')


plt.savefig('EGOsolution'+'.svg')
plt.show()

### Plot param
"""
ylabel = [r'$D_1$', r'$D_2$', r'$D_3$', r'$D_4$', r'$\rho_1$', r'$\rho_2$', r'$\rho_3$',]

for i in range(7):#dataset.query_points.get_shape()[1]-1):
    plt.figure(i+1)
    plt.plot(ind_lowfi[:],lowfi_points[:,i],'o--', label="LF",color="blue");
    plt.plot(ind_highfi[:],highfi_points[:,i], 'o--', label="HF",color="red");

    plt.fill_between([0,DOE_bf], [202,202], color='blue', alpha=0.1)
    plt.fill_between([DOE_bf,DOE_bf+DOE_hf], [202,202], color='red', alpha=0.1)

    plt.xlabel("evaluation number");
    plt.ylabel(ylabel[i]);
    plt.ylim(80,202)
    plt.xlim(0,)
    plt.legend();
    plt.savefig(f'variables_{i+1}.svg')
"""
#ylabel = [r'$D_1$', r'$D_2$', r'$D_3$', r'$D_4$', r'$\rho_1$', r'$\rho_2$', r'$\rho_3$',]
# for i in range(dataset.query_points.get_shape()[1]-1):
#     plt.figure(i+2+7)
#     plt.plot(lowfi_points[:,i],tf.gather(dataset.observations, ind_lowfi)[:],'o--', label="LF",color="blue");
#     plt.plot(highfi_points[:,i],tf.gather(dataset.observations, ind_highfi)[:], 'o--', label="HF",color="red");
#
#     plt.xlabel("evaluation number");
#     plt.ylabel(ylabel[i]);
#     #plt.ylim(80,202)
#     #plt.xlim(0,)
#     plt.legend();
#     plt.savefig(f'variables_{i}.svg')
