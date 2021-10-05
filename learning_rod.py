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
    'rew_mode': 'approx_diff',  # Mode of rewards ('forces', 'abs_forces', 'primitive', 'WLU', 'approx_diff' or 'classic')
    'close_pen': 0,             # Prefactor for closeness penalty (closenes to other particles)
    'prox_rew': 0,              # Reward prefactor for proximity reward (prox. to rod)
    'r_rew_fact': 0,            # Reward prefactor for rotation rewards for rewards based on forces
    'p_rew_fact': 5,            # Reward prefactor for pushing in long difection
    'rew_cutoff': 60,           # Cutoff for the primitive/WLU rewards
    'flag_fix_or': 0,           # Determines, if the direction to move the rod in mode 6 is fixed to the original rod orientation or not.
    'trans_dist': 100,          # distance, over which the rod should be transportet in mode 7
    'sparse_rew': False,        # gives only one, random particle a reward every step
    'n_rew_frames': 1,          # number of frames one particle is rewarded in the sparse_rew==true mode

    # for primitive reward
    'prim_rew_mode': 'close',   # 'close' or 'touch' determining, whether rewards are given in case of touching or closeness

    # for diff Reward
    'WLU_prefact': 10000,       # Prefactor for WLU rewards (1e4 is good for rotation)
    'WLU_mode': 'non_ex',       # 'non_ex', 'passive' or 'switch' as clamping parameter
    'WLU_noise' : 'mixed',      # noise in determination of performance and hypPerformance for WLU Reward ('on', 'off', 'mixed', 'no' or 'ideal')

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
    'skew': False,              # Flag for if the initialization positions are all on one side
    'Dt': 0.014,                # translational diffusion coefficient
    'Dr': 1.0 / 350.0,          # rotational diffusion coefficient

    # Rod
    'n_rod': 60,                # must be even!
    'part_size_rod': 0.01,
    'mu_K': 1.8,
    'l_rod': 96,                # length of the rod
    # 'distRod': 1.6,           # is calculated in environments/rod.py from the size and the number of rod particles
    'ext_rod': 1.,
    'cen_rod': 1.,
    'fr_rod': 10,               # friction of the rod determining, how easily the particles can move it (10 is close to exp.)

    # For the MD part of the simulation
    'train_ep': 100,            # number of episodes conducted during the whole training (replaces n_MD)
    'eval_ep': 3,               # number of evaluation episodes doen in the end without further training

    'train_frames': 1000,       # number of simulation frames done in one training episode; each step covers int_steps * dt in time.
    'eval_frames': 1000,        # number of simulation frames done in one evaluation episode
    'train_pause': 128,         # number of simulation frames, after which there is one step of training

    'int_steps': 20,            # number of times, the integration is performed in each simulation step
    'dt': 0.1                   # time step of integration in simulations
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
    selected_params = dict(zip(
        job_parameters.keys(),
        [vals.flat[task_id - 1] for vals in np.meshgrid(*job_parameters.values())]
    ))

    # Constructs the folder name for the task from the relevant parameters
    data_dir = os.path.join(
        job_dir,
        '_'.join([key + str(val) for key, val in selected_params.items()])
    )

    do_task(selected_params, data_dir)



def do_task(selected_params, data_dir):
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
    parameters.update(selected_params)

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

    if parameters['rew_mode'] == 'approx_diff':
        parameters['approx_flag'] = True

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



def do_episode_batch(agent, parameters, data_dir, name, n_episodes, n_step_ep, *, rec_traj=False, train_agent=False, debugging=False):

    # n_step_ep is the number of simulation steps (observables -> actions -> evolved environment) in one episode
    # n_episodes is the number of episodes to be conducted in this batch

    # Set up the data storage file in h5 format
    store_file = h5py.File(os.path.join(data_dir, name + '.h5'), 'w')

    rewards = store_file.create_dataset('/rewards', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    rod_or = store_file.create_dataset('/rod_or', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    rod_cm = store_file.create_dataset('/rod_cm', (n_episodes,n_step_ep,2), dtype='f4', compression='gzip') # rod_com[:,:,0] is x-component and rod_com[:,:,1] is y-component
    entropies = store_file.create_dataset('/entropies', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    values = store_file.create_dataset('/values', (n_episodes,n_step_ep), dtype='f4', compression='gzip')

    for i_ep in range(0, n_episodes):

        if rec_traj:
            rewards[i_ep,:], rod_or[i_ep,:], rod_cm[i_ep,:,:], entropies[i_ep,:], values[i_ep,:], target, particles, rod,\
            hyp_rod_ang, hyp_perf, perf, perf_rod_ang = \
                do_episode(agent, parameters, n_step_ep, rec_traj=rec_traj, train_agent=train_agent)

            rodName = 'traj{}/rod'.format(i_ep) # name of the dataset in the h5 file has to change for the trajectories
            partName = 'traj{}/particles'.format(i_ep)

            store_file.create_dataset(partName, compression='gzip', data=particles)
            store_file.create_dataset(rodName, compression='gzip', data=rod)

            # This is for looking at the hypothetical rods and performances, etc.
            if debugging:

                hyp_rods_name = 'traj{}/hypRods'.format(i_ep)          # hypothetical rods
                hyp_perf_name = 'traj{}/hypPers'.format(i_ep)          # hypothetical performances
                perfs_name = 'traj{}/perf'.format(i_ep)                # performance
                perf_rods_name = 'traj{}/perfRod'.format(i_ep)   # rod, from which the performance was determined

                store_file.create_dataset(hyp_rods_name, compression='gzip', data=hyp_rod_ang)
                store_file.create_dataset(hyp_perf_name, compression='gzip', data=hyp_perf)
                store_file.create_dataset(perfs_name, compression='gzip', data=perf)
                store_file.create_dataset(perf_rods_name, compression='gzip', data=perf_rod_ang)

        else:
            rewards[i_ep,:], rod_or[i_ep,:], rod_cm[i_ep,:,:], entropies[i_ep,:], values[i_ep,:], target = \
                do_episode(agent, parameters, n_step_ep, rec_traj=rec_traj, train_agent=train_agent)

        # In the case of the transportation problem, the target is saved
        if parameters['mode'] == 7:
                tar_name = 'traj{}/target'.format(i_ep)
                store_file.create_dataset(tar_name, compression='gzip', data=target)

    store_file.close()



def do_episode(agent, parameters, n_step_ep, *, rec_traj=False, train_agent=False):

    # Initializing the data arrays
    mean_rew = np.zeros((n_step_ep), dtype='f4')
    rod_or = np.zeros((n_step_ep), dtype='f4')
    rod_cm = np.zeros((1,n_step_ep,2), dtype='f4')
    mean_ent = np.zeros((n_step_ep), dtype='f4')
    mean_val = np.zeros((n_step_ep), dtype='f4')

    if rec_traj:
        # Making arrays for the rod-particle positions and the particle data
        particles = np.full((n_step_ep, 5, parameters['N']), fill_value=np.nan) # order: X, Y, Theta, actions, rewards
        rod = np.full((n_step_ep, 2, parameters['n_rod']), fill_value=np.nan) # order: X, Y

        # for debugging the hypothetical performance, hyp. rods, performace and the rod, from which
        # the performance was determined are saved, too
        hyp_rod_ang = np.zeros((parameters['N'], n_step_ep), dtype='f4') # just the angle is saved (self.N x n_step_ep values)
        hyp_perf = np.zeros((parameters['N'], n_step_ep), dtype='f4') # (self.N x n_step_ep values)
        perf = np.zeros((n_step_ep), dtype='f4') # the overall performance in one timestep
        perf_rod_ang = np.zeros((n_step_ep), dtype='f4') # the rod, from which perf was determined

    # Initialize the environment
    environment = MD_ROD(**parameters)
    obs, rewards = environment.get_obs_rewards() # gets first obs and rewards

    # In the case of rew_mode == 'approx_diff', the approximated reward for all
    # paricles is subtracted
    if parameters['rew_mode'] == 'approx_diff':
        rewards = rewards - agent.approx(obs).numpy().reshape(-1)

    # Initialize the agent
    agent.initialize(obs)

    # Real simulation loop
    for step in range(n_step_ep):

        # Agent decides actions from the observables
        actions, logp = agent.get_actions()

        # The environment is updated according to the selected actions
        obs, rewards, rod_theta, rod_com = environment.evolve_MD(actions)

        # In the case of rew_mode == 'approx_diff', the approximated reward for all
        # paricles is subtracted
        if parameters['rew_mode'] == 'approx_diff':
            rewards = rewards - agent.approx(obs).numpy().reshape(-1)

        # Add the environment response to the knowledge od the agent
        values = agent.add_environment_response(environment.lost, obs, rewards)

        # Train the agent
        if train_agent and (step+1) % parameters['train_pause'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(obs)

        # Save the important information in the h5 file
        mean_rew[step] = np.mean(rewards)
        rod_or[step] = rod_theta
        rod_cm[0,step,:] = rod_com
        mean_ent[step] = np.mean(scipy.stats.entropy(np.exp(logp), base=agent.n_actions, axis=1))
        mean_val[step] = np.mean(values)

        if rec_traj:
            # Save the particle positions, actions and rewards and the rod-particle poositions
            particles[step, 3, :] = actions
            particles[step, 0:3, :] = environment.particles.transpose()
            particles[step, 4, :] = rewards
            rod[step, :, :] = environment.rod.transpose()

            # for debugging the hypothetical performance, hyp. rods, performace and the rod, from which
            # the performance was determined are saved, too
            hyp_rod_ang[:,step] = environment.hyp_rod_ang
            hyp_perf[:,step] = environment.hyp_perf
            perf[step] = environment.performance
            perf_rod_ang[step] = environment.perf_rod_ang


    agent.finish_episode()
    if not train_agent:
        agent.reset_memory()

    if rec_traj:
        return mean_rew, rod_or, rod_cm, mean_ent, mean_val, environment.target, particles, rod, hyp_rod_ang, hyp_perf, perf, perf_rod_ang
    else:
        return mean_rew, rod_or, rod_cm, mean_ent, mean_val, environment.target





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
