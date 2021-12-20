#! /bin/python3

import sys
import json
import os
import h5py
import numpy as np
import tensorflow as tf
import environments.rod
import firstrl


### Check User Input

# assert len(sys.argv) > 1, 'You need to specify a directory'
run_dir = '/data/scc/veit-lorenz.heuthe/friction_sweep/2021-08-24/WLU_modenon_ex_WLU_noisemixed_l_rod100_fr_rod50_rep2' # sys.argv[1]

assert os.path.exists(os.path.join(run_dir, 'parameters.json')), 'No parameter file found'
assert os.path.exists(os.path.join(run_dir, 'evaluation.h5')), 'No data found'


### Set up parameters

with open(os.path.join(run_dir, 'parameters.json'), 'r') as reader:
    parameters = json.load(reader)

# Make sure the agent is constructed from the right models
parameters['load_models'] = os.path.join(run_dir, 'model')

if not 'lr_pi' in parameters:
    parameters['lr_pi'] = parameters['lrPI']
    parameters['lr_v'] = parameters['lrV']


### generate data

env = environments.rod.MD_ROD(**parameters)
agent = firstrl.AgentActiveMatter(**parameters)

storage = h5py.File(os.path.join(run_dir, 'evaluation.h5'), 'r+')

rod = storage['/traj0/rod']
particles = storage['/traj0/particles']

print('Recalculating observables ...')

observables = np.zeros((particles.shape[0], particles.shape[2], parameters['n_obs']))

for i in range(particles.shape[0]):
    env.rod = rod[i].transpose()
    env.particles = particles[i, 0:3].transpose()
    observables[i], _ = env.get_obs_rewards(particles[i].transpose())

# not sure if this line is needed
agent.policy.trainable = False

print('Extracting features ...')

policy_extractor = tf.keras.Model(inputs=agent.policy.inputs, outputs=[layer.output for layer in agent.policy.layers])
critic_extractor = tf.keras.Model(inputs=agent.critic.inputs, outputs=[layer.output for layer in agent.critic.layers])

policy_features = list(zip(*[policy_extractor(obs) for obs in observables]))
critic_features = list(zip(*[critic_extractor(obs) for obs in observables]))


### save to h5 in a matlab friendly dimension order

print('Saving ...')

storage.create_dataset('/traj/observables', compression='gzip', data=np.transpose(observables, axes=(0,2,1)))
for i, features in enumerate(policy_features):
    storage.create_dataset('/traj/policy/layer'+str(i+1), compression='gzip', data=np.transpose(features, axes=(0,2,1)))
for i, features in enumerate(critic_features):
    storage.create_dataset('/traj/critic/layer'+str(i+1), compression='gzip', data=np.transpose(features, axes=(0,2,1)))

print('DONE')
