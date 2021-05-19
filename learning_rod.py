# Reworked analogous to Robert's learning_food.py, where
# executing an episode, step, etc. are all functions.
# TEST WITH MD
import numpy as np
import sys
import scipy
import time
import h5py
import os
import json
from scipy.stats import entropy as entropy

from environments.rod import MD_ROD
from firstrl import AgentActiveMatter



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
    'nActions': 4,
    'training_epochs': 50,

    # For Rewards
    'close_pen': 0, # Prefactor for closeness penalty
    'rotRewFact': 2, # Prefactor for rotation rewards for rewards based on forces
    'pushRewFact': 3,
    'rewMode': 'absForces', # Mode of rewards ('forces', 'absForces', 'primitive', 'primitiveTouch' or 'classic')
    'rewCutoff': 20, # float(sys.argv[1]), # 8, # Cutoff for the primitive rewards
    'mode': 6, # 3: normal rotation, 4: rotation in direction s, 2: directional pushing, 6:push along long direction

    # Particles
    'vel_act': 0.35,
    'vel_tor': 0.2, # Velocity during rotation (particles do not stand still)
    'N': 30,
    'torque': 25,
    'ss': 0.01,
    'obs_type': '1overR',
    'cones': 5,
    'cone_angle': 180,
    'flag_side': False,
    'flag_LOS': False,
    'startConfig': 'standard', # 'standard' or 'biased' or 'test_friction'

    # Rod
    'Nrod': 60, # must be even!
    'ss_rod': 0.01,
    'mu_K': 1,
    'sizeRod': 96,
    'distRod': 1.6,
    'ext_rod': 1.,
    'cen_rod': 1.,
    'massRod': 5, # "mass" of the rod determining, how easily the particles can move it (10 is close to exp.)

    # For the MD part of the simulation
    'nTrainEp': 100, # number of episodes conducted during the whole training (replaces n_MD)
    'nEvalEp': 3, # number of evaluation episodes doen in the end without further training

    'nStepEpTrain': 720, # number of simulation steps done in one training episode; each step covers nStepSim * dt in time.
    'nStepEpEval': 1000, #number of simulation steps done in one evaluation episode
    'stepsTrain': 128, # number of simulation steps, after which there is a training update

    'nStepSim': 20, # number of times, the integration is performed in each simulation step
    'dt': 0.1, # time step of integration in simulations
    'skew': False,
    'size': 100
}



def do_array_task(task_id, job_dir): # Copied from Robert
    '''
    This takes the qsub task_id and with that produces a set of parameters from the json file in job_dir.
    '''

    # parameter ranges are stored in the job_dir
    with open(os.path.join(job_dir, 'parameters.json'), 'r') as reader:
        job_parameters = json.load(reader)

    # choose one set out of all possible parameter combinations
    # (task_id's start at 1 !!)
    selectedParameters = dict(zip(
        job_parameters.keys(),
        [vals.flat[task_id - 1] for vals in np.meshgrid(*job_parameters.values())]
    ))

    # Constructs the folder name for the task from the relevant parameters
    dataDir = os.path.join(
        job_dir,
        '_'.join([key + str(val) for key, val in selectedParameters.items()])
    )

    do_task(selectedParameters, dataDir)



def do_task(selectedParameters, dataDir):
    '''
    This takes the set of parameters selected by do_array_task and does two batches
    of simulations: First nTrainEp training episodes and then nEvalEp evaluation episodes.#
    The results are saved individually for the two batches in h5 files in a folder
    'dataDir', carrying the important parameter (the one that is changed) as its name.
    '''

    # Make the data directory
    os.makedirs(dataDir, exist_ok=True)

    # Use the default parameters but update the specified ones
    parameters = default_parameters.copy()
    parameters.update(selectedParameters)

    # Initializing the agent. It's the same agent throughout all the batches in one task.
    agent = AgentActiveMatter(**parameters)
    agent.save_models(os.path.join(dataDir, 'model'))

    # Save the used parameters to a json file for tracability
    with open(os.path.join(dataDir, 'parameters.json'), 'w', encoding='utf8') as paramFile:
        json.dump(parameters, paramFile, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # Now there is training for nTrainEp episodes (training batch)
    do_episode_batch(agent, parameters, dataDir, 'training', parameters['nTrainEp'], parameters['nStepEpTrain'], recordTraj=False, trainAgent=True)

    # Training is done at this point
    agent.save_models(os.path.join(dataDir, 'model'))

    # And then evaluation for nEvalEp episodes (evaluation batch)
    do_episode_batch(agent, parameters, dataDir, 'evaluation', parameters['nEvalEp'], parameters['nStepEpEval'], recordTraj=True, trainAgent=False)



def do_episode_batch(agent, parameters, dataDir, name, nEpisodes, nStepEp, *, recordTraj=False, trainAgent=False):

    # nStepEp is the number of simulation steps (observables -> actions -> evolved environment) in one episode
    # nEpisodes is the number of episodes to be conducted in this batch

    # Set up the data storage file in h5 format
    storFile = h5py.File(os.path.join(dataDir, name + '.h5'), 'w')

    rewards = storFile.create_dataset('/rewards', (nEpisodes,nStepEp), dtype='f4', compression='gzip')
    rodOr = storFile.create_dataset('/rodOr', (nEpisodes,nStepEp), dtype='f4', compression='gzip')
    rodCM = storFile.create_dataset('/rodCM', (nEpisodes,nStepEp,2), dtype='f4', compression='gzip') # rodCoM[:,:,0] is x-component and rodCoM[:,:,1] is y-component
    entropies = storFile.create_dataset('/entropies', (nEpisodes,nStepEp), dtype='f4', compression='gzip')
    values = storFile.create_dataset('/values', (nEpisodes,nStepEp), dtype='f4', compression='gzip')

    for iEp in range(0, nEpisodes):

        if recordTraj:
            rewards[iEp,:], rodOr[iEp,:], rodCM[iEp,:,:], entropies[iEp,:], values[iEp,:] , particles, rod = \
                do_episode(agent, parameters, nStepEp, recordTraj=recordTraj, trainAgent=trainAgent)

            rodName = 'traj{}/rod'.format(iEp) # name of the dataset in the h5 file has to change for the trajectories
            partName = 'traj{}/particles'.format(iEp)

            storFile.create_dataset(partName, compression='gzip', data=particles)
            storFile.create_dataset(rodName, compression='gzip', data=rod)

        else:
            rewards[iEp,:], rodOr[iEp,:], entropies[iEp,:], values[iEp,:] = \
                do_episode(agent, parameters, nStepEp, recordTraj=recordTraj, trainAgent=trainAgent)

    storFile.close()



def do_episode(agent, parameters, nStepEp, *, recordTraj=False, trainAgent=False):

    # Initializing the data arrays
    meanRew = np.zeros((nStepEp), dtype='f4')
    rodOr = np.zeros((nStepEp), dtype='f4')
    rodCM = np.zeros((nStepEp,1,2), dtype='f4')
    meanEntr = np.zeros((nStepEp), dtype='f4')
    meanVal = np.zeros((nStepEp), dtype='f4')

    if recordTraj:
        # Making arrays for the rod-particle positions and the particle data
        particles = np.full((nStepEp, 5, parameters['N']), fill_value=np.nan) # order: X, Y, Theta, actions, rewards
        rod = np.full((nStepEp, 2, parameters['Nrod']), fill_value=np.nan) # order: X, Y

    # Initialize the environment
    environment = MD_ROD(**parameters)
    obs, rewards = environment.get_obs_rewards() # gets first obs and rewards

    # Initialize the agent
    agent.initialize(obs)

    # Real simulation loop
    for step in range(nStepEp):

        # Agent decides actions from the observables
        actions, logp = agent.get_actions()

        # The environment is updated according to the selected actions
        obs, rewards, rodTheta, rodCoM = environment.evolve_MD(actions)

        # Add the environment response to the knowledge od the agent
        values = agent.add_environment_response([], obs, rewards)

        # Train the agent
        if trainAgent and (step+1) % parameters['stepsTrain'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(obs)

        # Save the important information in the h5 file
        meanRew[step] = np.mean(rewards)
        rodOr[step] = rodTheta
        rodCM[step,1,:] = rodCoM
        meanEntr[step] = np.mean(scipy.stats.entropy(np.exp(logp), base=agent.nActions, axis=1))
        meanVal[step] = np.mean(values)

        if recordTraj:
            # Save the particle positions, actions and rewards and the rod-particle poositions
            particles[step, 3, :] = actions
            particles[step, 0:3, :] = environment.particles.transpose()
            particles[step, 4, :] = rewards
            rod[step, :, :] = environment.rod.transpose()

    agent.finish_episode()
    if not trainAgent:
        agent.reset_memory()

    if not recordTraj:
        return meanRew, rodOr, meanEntr, meanVal
    else:
        return meanRew, rodOr, meanEntr, meanVal, particles, rod



class NumpyEncoder(json.JSONEncoder): # Copied from Robert
    """Helps parsing integer parameter ranges

    See https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    """
    def default(self, obj):  # pylint: disable=method-hidden
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)
