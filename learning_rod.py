# Reworked analogous to Robert's learning_food.py, where
# executing an episode, step, etc. are all functions.
# TEST WITH MD
import numpy as np
import scipy
import h5py
import os
import json
from scipy.stats import entropy as entropy

from environments.rod import MD_ROD
from firstrl import AgentActiveMatter



default_parameters = {

    # RL Agent
    'n_obs': 10,
    'n_actions': 4,
    'en_coeff': 0.01,
    'CL': 0.03,
    'gamma': 0.95,
    'lam': 0.97,
    'lr_pi': 0.0005,
    'lr_v': 0.001,
    'target_kl': 0.02,
    'model_structure': [(32, 'relu'),(16, 'relu'),(16, 'relu')],
    'training_epochs': 50,
    'load_models': None,

    # For Rewards
    'mode': 3,                  # 3: normal rotation, 4: rotation in direction s, 2: directional pushing, 6:push along long direction, 7: Rod transportation
    'rew_mode': 'WLU',          # Mode of rewards ('forces', 'absForces', 'primitive', 'primitiveTouch', 'WLU' or 'classic')
    'close_pen': 0,             # Prefactor for closeness penalty (closenes to other particles)
    'prox_rew': 0,              # Reward prefactor for proximity reward (prox. to rod)
    'r_rew_fact': 2,            # Reward prefactor for rotation rewards for rewards based on forces
    'p_rew_fact': 5,            # Reward prefactor for pushing in long difection
    'rew_cutoff': 60,           # Cutoff for the primitive/WLU rewards
    'flag_fix_or': 0,           # Determines, if the direction to move the rod in mode 6 is fixed to the original rod orientation or not.
    'trans_dist': 100,          # distance, over which the rod should be transportet in mode 7
    'sparse_rew': False,        # gives only one, random particle a reward every step
    'n_rew_frames': 1,          # number of frames one particle is rewarded in the sparse_rew==true mode

    # for diff Reward
    'WLU_prefact': 10000,       # Prefactor for WLU rewards (1e4 is good for rotation)
    'WLU_mode': 'non_ex',       # 'non_ex' for non-existing particles or 'passive' for passive particles for determining the hypPerformance ('switch' for combi)
    'WLU_noise' : 'ideal',      # noise in determination of performance and hypPerformance for WLU Reward ('on', 'off', 'mixed', 'no' or 'ideal')

    # Particles
    'vel_act': 0.45,            # 0.35,
    'vel_tor': 0.2,             # Velocity during rotation (particles do not stand still)
    'N': 30,
    'torque': 25,
    'part_size': 0.01,
    'obs_type': '1overR',
    'cones': 5,
    'cone_angle': 180,
    'flag_side': False,
    'flag_LOS': False,
    'start_conf': 'standard',   # 'standard' or 'biased' or 'test_friction' or 'transportation'
    'Dt': 0.014,                # translational diffusion coefficient
    'Dr': 1.0 / 350.0,          # rotational diffusion coefficient

    # Rod
    'n_rod': 60,                 # must be even!
    'part_size_rod': 0.01,
    'mu_K': 1.8,
    'l_rod': 96,                # length of the rod
    # 'distRod': 1.6,           # is calculated in environments/rod.py from the size and the number of rod particles
    'ext_rod': 1.,
    'cen_rod': 1.,
    'fr_rod': 1,                # friction of the rod determining, how easily the particles can move it (10 is close to exp.)

    # For the MD part of the simulation
    'train_ep': 100,            # number of episodes conducted during the whole training (replaces n_MD)
    'eval_ep': 3,               # number of evaluation episodes doen in the end without further training

    'train_frames': 1000,       # number of simulation frames done in one training episode; each step covers int_steps * dt in time.
    'eval_frames': 1000,        # number of simulation frames done in one evaluation episode
    'train_pause': 128,         # number of simulation frames, after which there is one step of training

    'int_steps': 20,            # number of times, the integration is performed in each simulation step
    'dt': 0.1,                  # time step of integration in simulations
    'skew': False               # Flag for if the initialization positions are all on one side
}



def do_array_task(task_id, job_dir): # Copied from Robert
    '''
    This takes the qsub task_id and with that produces a set of parameters from the json file in job_dir.
    This is then fed into do_task
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

    for rep in range(1, selectedParameters['nRep'] + 1):

        # Constructs the folder name for the task from the relevant parameters
        data_dir = os.path.join(
            job_dir,
            '_'.join([key + str(val) for key, val in selectedParameters.items()]),
            'rep_{}'.format(rep)
        )

        do_task(selectedParameters, data_dir)



def do_task(selectedParameters, data_dir):
    '''
    This takes the set of parameters selected by do_array_task and does two batches
    of simulations: First train_ep training episodes and then eval_ep evaluation episodes.#
    The results are saved individually for the two batches in h5 files in a folder
    'data_dir', carrying the important parameter (the one that is changed) as its name.
    '''

    # Make the data directory
    os.makedirs(data_dir, exist_ok=True)

    # Use the default parameters but update the specified ones
    parameters = default_parameters.copy()
    parameters.update(selectedParameters)

    # Make sure, that the input dimension and the start_conf is ok (self-consistency),
    # so one does not have to specify the parameters dependant on the mode.
    if parameters['mode'] == 3:
        parameters['n_obs'] = 2 * parameters['cones']
    elif parameters['mode'] == 6:
        parameters['n_obs'] = 2 * parameters['cones'] + 1
        parameters['start_conf'] = 'standard'
    elif parameters['mode'] == 7:
        parameters['n_obs'] = 3 * parameters['cones']
        parameters['start_conf'] = 'transportation'
        parameters['rew_mode'] = 'WLU'

    # Initializing the agent. It's the same agent throughout all the batches in one task.
    agent = AgentActiveMatter(**parameters)
    agent.save_models(os.path.join(data_dir, 'model'))

    # Save the used parameters to a json file for tracability
    with open(os.path.join(data_dir, 'parameters.json'), 'w', encoding='utf8') as param_file:
        json.dump(parameters, param_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # Now there is training for train_ep episodes (training batch)
    do_episode_batch(agent, parameters, data_dir, 'training', parameters['train_ep'], parameters['train_frames'], rec_traj=False, train_agent=True, debugging=False)

    # Training is done at this point
    agent.save_models(os.path.join(data_dir, 'model'))

    # And then evaluation for eval_ep episodes (evaluation batch)
    do_episode_batch(agent, parameters, data_dir, 'evaluation', parameters['eval_ep'], parameters['eval_frames'], rec_traj=True, train_agent=False, debugging=True)



def do_episode_batch(agent, parameters, data_dir, name, nEpisodes, nStepEp, *, rec_traj=False, train_agent=False, debugging=False):

    # nStepEp is the number of simulation steps (observables -> actions -> evolved environment) in one episode
    # nEpisodes is the number of episodes to be conducted in this batch

    # Set up the data storage file in h5 format
    storFile = h5py.File(os.path.join(data_dir, name + '.h5'), 'w')

    rewards = storFile.create_dataset('/rewards', (nEpisodes,nStepEp), dtype='f4', compression='gzip')
    rodOr = storFile.create_dataset('/rodOr', (nEpisodes,nStepEp), dtype='f4', compression='gzip')
    rodCM = storFile.create_dataset('/rodCM', (nEpisodes,nStepEp,2), dtype='f4', compression='gzip') # rodCoM[:,:,0] is x-component and rodCoM[:,:,1] is y-component
    entropies = storFile.create_dataset('/entropies', (nEpisodes,nStepEp), dtype='f4', compression='gzip')
    values = storFile.create_dataset('/values', (nEpisodes,nStepEp), dtype='f4', compression='gzip')

    for iEp in range(0, nEpisodes):

        if rec_traj:
            rewards[iEp,:], rodOr[iEp,:], rodCM[iEp,:,:], entropies[iEp,:], values[iEp,:], target, particles, rod,\
            hypRodAng, hypPerformances, perf, perfRodAng = \
                do_episode(iEp, agent, parameters, nStepEp, rec_traj=rec_traj, train_agent=train_agent)

            rodName = 'traj{}/rod'.format(iEp) # name of the dataset in the h5 file has to change for the trajectories
            partName = 'traj{}/particles'.format(iEp)

            storFile.create_dataset(partName, compression='gzip', data=particles)
            storFile.create_dataset(rodName, compression='gzip', data=rod)

            # This is for looking at the hypothetical rods and performances, etc.
            if debugging:

                hypRodsName = 'traj{}/hypRods'.format(iEp)          # hypothetical rods
                hypPersName = 'traj{}/hypPers'.format(iEp)          # hypothetical performances
                perfName = 'traj{}/perf'.format(iEp)                # performance
                perfRodName = 'traj{}/perfRod'.format(iEp)   # rod, from which the performance was determined

                storFile.create_dataset(hypRodsName, compression='gzip', data=hypRodAng)
                storFile.create_dataset(hypPersName, compression='gzip', data=hypPerformances)
                storFile.create_dataset(perfName, compression='gzip', data=perf)
                storFile.create_dataset(perfRodName, compression='gzip', data=perfRodAng)

        else:
            rewards[iEp,:], rodOr[iEp,:], rodCM[iEp,:,:], entropies[iEp,:], values[iEp,:], target = \
                do_episode(iEp, agent, parameters, nStepEp, rec_traj=rec_traj, train_agent=train_agent)

        # In the case of the transportation problem, the target is saved
        if parameters['mode'] == 7:
                targetName = 'traj{}/target'.format(iEp)
                storFile.create_dataset(targetName, compression='gzip', data=target)

    storFile.close()



def do_episode(iEp, agent, parameters, nStepEp, *, rec_traj=False, train_agent=False):

    # Initializing the data arrays
    meanRew = np.zeros((nStepEp), dtype='f4')
    rodOr = np.zeros((nStepEp), dtype='f4')
    rodCM = np.zeros((1,nStepEp,2), dtype='f4')
    meanEntr = np.zeros((nStepEp), dtype='f4')
    meanVal = np.zeros((nStepEp), dtype='f4')

    if rec_traj:
        # Making arrays for the rod-particle positions and the particle data
        particles = np.full((nStepEp, 5, parameters['N']), fill_value=np.nan) # order: X, Y, Theta, actions, rewards
        rod = np.full((nStepEp, 2, parameters['n_rod']), fill_value=np.nan) # order: X, Y

        # for debugging the hypothetical performance, hyp. rods, performace and the rod, from which
        # the performance was determined are saved, too
        hypRodAng = np.zeros((parameters['N'], nStepEp), dtype='f4') # just the angle is saved (self.N x nStepEp values)
        hypPerformances = np.zeros((parameters['N'], nStepEp), dtype='f4') # (self.N x nStepEp values)
        perf = np.zeros((nStepEp), dtype='f4') # the overall performance in one timestep
        perfRodAng = np.zeros((nStepEp), dtype='f4') # the rod, from which perf was determined

    # Initialize the environment
    environment = MD_ROD(**parameters)
    obs, rewards = environment.get_obs_rewards(iEp) # gets first obs and rewards

    # Initialize the agent
    agent.initialize(obs)

    # Real simulation loop
    for step in range(nStepEp):

        # ZZZ Just for debugging
        if step > 1:
            old_rewards = new_rewards

        # Agent decides actions from the observables
        actions, logp = agent.get_actions()

        # The environment is updated according to the selected actions
        obs, rewards, rodTheta, rodCoM = environment.evolve_MD(iEp, actions)

        # ZZZ For debugging: if the rewards flicker too much, there is a hold point
        new_rewards = rewards
        if step > 500:
            if  iEp > 3: # (abs(sum(new_rewards) - sum(old_rewards)) > 7) and
                zzz = 1

        # Add the environment response to the knowledge od the agent
        values = agent.add_environment_response(environment.lost, obs, rewards)

        # Train the agent
        if train_agent and (step+1) % parameters['train_pause'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(obs)

        # Save the important information in the h5 file
        meanRew[step] = np.mean(rewards)
        rodOr[step] = rodTheta
        rodCM[0,step,:] = rodCoM
        meanEntr[step] = np.mean(scipy.stats.entropy(np.exp(logp), base=agent.n_actions, axis=1))
        meanVal[step] = np.mean(values)

        if rec_traj:
            # Save the particle positions, actions and rewards and the rod-particle poositions
            particles[step, 3, :] = actions
            particles[step, 0:3, :] = environment.particles.transpose()
            particles[step, 4, :] = rewards
            rod[step, :, :] = environment.rod.transpose()

            # for debugging the hypothetical performance, hyp. rods, performace and the rod, from which
            # the performance was determined are saved, too
            hypRodAng[:,step] = environment.hypRodAng
            hypPerformances[:,step] = environment.hypPerformances
            perf[step] = environment.performance
            perfRodAng[step] = environment.perfRodAng


    agent.finish_episode()
    if not train_agent:
        agent.reset_memory()

    if rec_traj:
        return meanRew, rodOr, rodCM, meanEntr, meanVal, environment.target, particles, rod, hypRodAng, hypPerformances, perf, perfRodAng
    else:
        return meanRew, rodOr, rodCM, meanEntr, meanVal, environment.target





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
