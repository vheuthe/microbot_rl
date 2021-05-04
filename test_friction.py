# This is for testing the friction by pushing
# two particles along the rod and evaluating,
# how fast the rod moves.

import numpy as np
import os
import scipy
import time
from scipy.stats import entropy as entropy

from environments.rod import MD_ROD

# ----------------------------------------------------------
# Defining some parameters
default_parameters = {

    # RL Agent
    'input_dim': 10,
    'output_dim': 4,
    'en_coeff': 0.01,
    'CL': 0.03,
    'gamma': 0.95,
    'lam': 0.97,
    'lrPI': 0.0005,
    'lrV': 0.001,
    'target_kl': 0.02,
    'model_structure': [(32, 'relu'),(16, 'relu'),(16, 'relu')],

    # For Rewards
    'close_pen': 0.6, # Prefactor for closeness penalty

    # Particles
    'vel_act': 0.35,
    'vel_tor': 0.2,
    'N': 2,
    'torque': 25,
    'ss': 0.01,
    'obs_type': '1overR',
    'conse': 5,
    'cone_angle': 180,
    'flag_side': False,
    'flag_LOS': False,
    'Dr': 0,
    'Dt': 0,

    # Rod
    'ss_rod': 0.01,
    'mu_K': 1.2,
    'sizeRod': 96,
    'distRod': 1.6,
    'ext_rod': 1.,
    'cen_rod': 1.,

    # For MD part of simulation
    'steps_update': 128,
    'n_MD': 1,
    'total_time': 7200,
    'step_time': 5,
    'dt': 0.1,
    'start_MD': 0,
    'mode': 3, # 3: normal rotation, 4: rotation in direction s, 2: directional pushin, 6:push along long direction
    'skew': False,
    'size': 100,
    'traj': True
}

parameters = default_parameters

def from_policy_to_actions(Pi):
    '''
    takes distribution of log probabilities over discrete set of actions
    and gives out one randomly, after normalization
    MUST RETURN only one value: index of action!
    '''
    action=np.random.choice(4,p=Pi)
    return action

mu_K_range = [1, 2]

# --------------------------------------------------------
# Running the simulations

for mu_K in mu_K_range:

    parameters['mu_K'] = mu_K

    data_path = '/mnt/c/Users/Veit/Documents/PhD/SimData/test_friction/mu_K_{}'.format(mu_K)

    os.system('mkdir -p {}'.format(data_path))

    n_max_steps = int(parameters["total_time"]/parameters["step_time"])

    md = MD_ROD(
        index=1, **parameters,
        data_path=data_path)

    # Do the test-episode
    for step in range(n_max_steps):
        md.evolve_MD([1,1])
        md.print_xyz_actions([1,1], [[0.5, 0.5, 0.5, 0.5],[0.5, 0.5, 0.5, 0.5]])