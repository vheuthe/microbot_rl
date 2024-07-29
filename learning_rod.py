# Reworked analogous to Robert's learning_food.py, where
# executing an episode, step, etc. are all functions.
# TEST WITH MD
import os
import json
import numpy as np
import scipy
import h5py
import time

from environments.rod import MD_ROD
from firstrl import AgentActiveMatter


default_parameters = {

    # RL Agent
    'n_actions': 4,
    'en_coeff': 0.01,
    'CL': 0.03,
    'gamma': 0.997,
    'lam': 0.97,
    'lr_pi': 0.0005,
    'lr_v': 0.00004,
    'target_kl': 0.02,
    'model_structure': [(32, 'relu'),(16, 'relu'),(16, 'relu')],
    'actor_epochs': 50,
    'critic_epochs': 1,
    'load_models': None,

    # Vision
    'obs_type': '1overR',
    'cones': 5,
    'cone_angle': np.pi,
    'flag_side': False,
    'flag_LOS': False,

    # For Rewards
    'mode': 7,                  # 3: normal rotation, 4: rotation in direction s, 2: directional pushing, 6:push along long direction, 7: Rod transportation
    'rew_mode': 'CR',          # Mode of rewards ('forces', 'abs_forces', 'team', 'CR' or 'classic')
    'close_pen': 0,             # Prefactor for closeness penalty (closenes to other particles)
    'prox_rew': 0,              # Reward prefactor for proximity reward (prox. to rod)
    'r_rew_fact': 100,          # Reward prefactor for rotation rewards for rewards based on forces
    'rew_cutoff': 60,           # Cutoff for the team/CR rewards
    'flag_fix_or': 0,           # Determines, if the direction to move the rod in mode 6 is fixed to the original rod orientation or not.
    'trans_dist': 50,           # distance, over which the rod should be transportet in mode 7
    'trans_dist_ramp': False,   # ramp up the trans_dist from 10 to trans_dist
    'max_trans_dist': 100,      # maximum distance ofer which to transport the rod with ramping trans_dist
    'target_tol': 120,          # allowed residual cummulative distance between target and rod for completion of the task
    'final_rew': 1000,          # the reward upon achieved task for truely episodic learning
    'bootstrap': True,          # flag for bootstrapping in episodic tasks
    'cost_iso_rew': False,      # cost instead of reward in episodic task mode 7

    # for team reward
    'team_rew_mode': 'close',   # 'team', 'close' or 'touch' determining, whether rewards are given in case of touching or closeness

    # for diff Reward
    'CR_prefact': 10,          # Prefactor for CR rewards (1e4 is good for rotation)
    'CR_mode': 'non_ex',       # 'non_ex', 'passive' or 'switch' as clamping parameter
    'CR_noise': 'mixed',       # noise in determination of performance and hypPerformance for CR Reward ('on', 'off', 'mixed', 'no' or 'ideal')
    'CR_rew_mode': 'touch',    # which particles are even considered ("touch"/"close")
    'CR_touch_rew': 0.1,       # Reward for touching in case of CR

    # Particles
    'vel_act': 7.3,             # Adjusted to optimize training time and match experiment
    'vel_tor': 2.3,             # Velocity during rotation (particles do not stand still)
    'vel_noise_fact': 0.5,      # factor for velocity noise of particles
    'rot_noise_fact': 0.5,      # factor for angular velocity noise of particles
    'N': 25,                    # Adjusted to optimize training time and match experiment
    'torque': 122,
    'part_size': 0.01,
    'start_conf': 'standard',   # 'standard' or 'biased' or 'test_friction' or 'transportation' (regarding the rod)
    'start_dist_scale': 1,      # scaling factor for the starting positions of the particles to initialize them far away
    'skew': False,              # Flag for if the initialization positions are all on one side
    'Dt': 0.014,                # translational diffusion coefficient
    'Dr': 1.0 / 350.0,          # rotational diffusion coefficient

    # Rod
    'n_rod': 60,                # must be even!
    'part_size_rod': 0.01,      # How big do particles see rod particles?
    'mu_K': 1.8,
    'l_rod': 96,                # length of the rod
    'ext_rod': 1.,              # how big are rod particles physically?
    'cen_rod': 1.,              # somehow alters the relative size of the different rod particles
    'fr_rod': 3,                # friction of the rod determining, how easily the particles can move it (10 is close to exp.)

    # Obstacles
    'use_obst': False,          # whether or not to include obstacles
    'obst_conf': "random",      # configuration of obstacles ('random' or 'wall')
    'obst_vision': False,       # whether or not the robots can see the obstacles

    # For the MD and training part of the simulation
    'episodic': True,           # flag for truely episodic training
    'train_ep': 200,            # number of episodes conducted during the whole training (replaces n_MD)
    'eval_ep': 5,               # number of evaluation episodes doen in the end without further training
    'episodic_eval': True,      # In case of episodic task, is the evaluation episodic or not?
    'termination_mode': "ind",  # changes the condition for when the task is achieved: either the "sum" or the "ind"ividual particles need to be smaller than achieved_dist
    'achieved_dist': 6,         # minimum distance for task achieved

    'train_frames': 370,        # number of simulation frames done in one training episode; each step covers int_steps * dt in time.
    'eval_frames': 1000,        # number of simulation frames done in one evaluation episode
    'train_pause': 60,          # number of simulation frames, after which there is one step of training
    'train_actor': True,        # Flag for whether or not to train the actor (useful for changing reward definitions while training)
    'reinitialize_critic': False,
    'record_traj': False,       # whether or not the full training trajectory is recorded

    'int_steps': 900,           # number of times, the integration is performed in each simulation step
    'dt': 0.001,                # time step of integration in simulations

    'parallelize_cr': False,    # whether or not the counterfactuals computation should be parallelized
    'n_processes': 1,           # number of processes in parallelization of counterfactual reward computation

    'eval_only': False    # flag for only running evaluation of runs without overwriting their training files
}


def do_array_task(task_id, job_dir): # Copied from Robert
    '''
    This takes the qsub task_id and with that produces a set of parameters from the json file in job_dir.
    This is then fed into do_task. The evaluation_only flag is for running the evaluation of a run that
    got stuck in training or something.
    '''

    # parameter ranges are stored in the job_dir
    with open(os.path.join(job_dir, 'parameters.json'), 'r', encoding="UTF-8") as reader:
        job_parameters = json.load(reader)

    # Choose one set out of all possible parameter combinations
    # (task_id's start at 1!!). Choose only the list instances in the
    # values of the parameters, otherwise the meshgrid gets too big
    # Screening parameters are the ones that are given as a list
    screening_params = {key: val for key, val in job_parameters.items() \
            if isinstance(val, list) and len(val) > 1 \
                and not (key in ("load_models", "model_structure"))}

    # Non screening parameters are the ones with single values
    non_screeening_params = {
        key: val for key, val in job_parameters.items() \
            if not isinstance(val, list) \
                or (isinstance(val, list) \
                    and len(val) == 1) \
                or key=="load_models" or key=="model_structure"}

    # Make sure that the non screening parameters are not lists
    non_screeening_params = {
        key: (val[0] if isinstance(val, list) else val)
        for key, val in non_screeening_params.items()
    }

    selected_params = dict(zip(
        screening_params.keys(),
        [values.flat[task_id - 1] for values in np.meshgrid(*screening_params.values())]
    ))
    selected_params.update(non_screeening_params)

    # Problems with the model structure: just copy the model structure from the job_params
    if "model_structure" in job_parameters.keys():
        selected_params["model_structure"] = job_parameters["model_structure"]

    # Constructs the folder name for the task from the relevant parameters
    # make sure to not use "load_models" for that, otherwise the paths get messed up)
    data_dir = os.path.join(
        job_dir,
        '_'.join([key + str(val) for key, val in selected_params.items() \
            if isinstance(job_parameters[key], list) and len(job_parameters[key]) > 1 and not (key=="load_models" or key=="model_structure")])
    )

    # Time the task execution
    start_time = time.time()
    do_task(selected_params, data_dir)
    end_time = time.time()
    print(f"Training took {end_time-start_time} seconds")


def do_task(selected_params, data_dir):
    '''
    This takes the set of parameters selected by do_array_task and does two batches
    of simulations: First train_ep training episodes and then eval_ep evaluation episodes.#
    The results are saved individually for the two batches in h5 files in a folder
    'data_dir', carrying the important parameter (the one that is changed) as its name.
    '''

    # Use the default parameters but update the specified ones
    parameters = default_parameters.copy()
    parameters.update(selected_params)

    # Make the data directory (don't, if its only evaluation)
    if not parameters["eval_only"]:
        os.makedirs(data_dir, exist_ok=True)

    # Make sure, that the input dimension and the start_conf is ok (self-consistency),
    # so one does not have to specify the parameters dependant on the mode.
    if parameters['mode'] == 2 or parameters['mode'] == 3:
        parameters['n_obs'] = 2 * parameters['cones']
        parameters['episodic'] = False
    elif parameters['mode'] == 4:
        parameters['n_obs'] = 2 * parameters['cones'] + 1
    elif parameters['mode'] == 6:
        parameters['n_obs'] = 2 * parameters['cones'] + 1
        parameters['start_conf'] = 'standard'
        parameters['episodic'] = False
    elif parameters['mode'] == 7:
        if parameters['start_conf'] not in ['transportation_long', 'transportation_trans', 'transp_1', 'transp_2']:
            parameters['start_conf'] = 'transportation'

        if parameters['rew_mode'] not in ['CR', 'CR_experiment']:
            parameters['rew_mode'] = 'CR'

        parameters['n_obs'] = 3 * parameters['cones']

    # Add observables for the obstacles
    if parameters['use_obst'] and parameters['obst_vision']:
        parameters['n_obs'] = parameters['n_obs'] + parameters['cones']
    else:
        parameters['obst_vision'] = False

    # Save the used parameters to a json file for tracability
    with open(os.path.join(data_dir, 'parameters.json'), 'w', encoding='utf8') as param_file:
        json.dump(parameters, param_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # If only evaluation should be ran, load the model from the job directory
    if parameters["eval_only"]:
        assert "model_policy" in os.listdir(data_dir) \
            and "model_critic" in os.listdir(data_dir) \
            and "training.h5" in os.listdir(data_dir), \
            f"No model to load in {data_dir}"
        assert "evaluation.h5" not in os.listdir(data_dir), \
            "Evaluation is already done"
        parameters["load_models"] = data_dir + "/model"

    # Initializing the agent. It's the same agent throughout all the batches in one task.
    agent = AgentActiveMatter(**parameters)
    if not parameters["eval_only"]:
        agent.save_models(os.path.join(data_dir, 'model'))

    # Now there is training for train_ep episodes (training batch)
    if parameters['train_ep'] > 0 and not parameters["eval_only"]:
        if parameters["episodic"]:
            do_episode_batch_episodic(
                agent, parameters, data_dir, 'training',
                parameters['train_ep'], parameters['train_frames'],
                rec_traj=parameters['record_traj'], train_agent=True, debugging=False)
        else:
            do_episode_batch(
                agent, parameters, data_dir, 'training',
                parameters['train_ep'], parameters['train_frames'],
                rec_traj=parameters['record_traj'], train_agent=True, debugging=False)

    # Training is done at this point
    agent.save_models(os.path.join(data_dir, 'model'))

    # And then evaluation for eval_ep episodes (evaluation batch)
    if parameters['eval_ep'] > 0:
        if parameters["episodic"] and parameters["episodic_eval"]:
            do_episode_batch_episodic(
                agent, parameters, data_dir, 'evaluation',
                parameters['eval_ep'], parameters['eval_frames'],
                rec_traj=True, train_agent=False, debugging=False)
        else:
            do_episode_batch(
                agent, parameters, data_dir, 'evaluation',
                parameters['eval_ep'], parameters['eval_frames'],
                rec_traj=True, train_agent=False, debugging=False)


def do_episode_batch(agent, parameters, data_dir, name, n_episodes, n_step_ep, *, rec_traj=False, train_agent=False, debugging=False):
    '''
    Executes a batch of episodes, either training or evaluation
    '''

    # n_step_ep is the number of simulation steps (observables -> actions -> evolved environment) in one episode
    # n_episodes is the number of episodes to be conducted in this batch

    # Special case: if the task is episodic, we are in the evaluation and ~episodic_eval,
    # the Flag for episodic is set to False here, so the episodes don't stop when the task
    # is achieved (this is somewhat hacky...)
    if parameters["episodic"] and name=='evaluation' and not parameters["episodic_eval"]:
        parameters["episodic"] = False

    # Set up the data storage file in h5 format
    store_file = h5py.File(os.path.join(data_dir, name + '.h5'), 'w')

    rewards = store_file.create_dataset('/rewards', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    rod_or = store_file.create_dataset('/rod_or', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    rod_cm = store_file.create_dataset('/rod_cm', (n_episodes,n_step_ep,2), dtype='f4', compression='gzip') # rod_com[:,:,0] is x-component and rod_com[:,:,1] is y-component
    entropies = store_file.create_dataset('/entropies', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    values = store_file.create_dataset('/values', (n_episodes,n_step_ep), dtype='f4', compression='gzip')
    elapsed_times = store_file.create_dataset('elapsed_times', (n_episodes,1), dtype='f4', compression='gzip')

    for i_ep in range(0, n_episodes):

        # Get the start time
        ep_start_time = time.time()

        # Do the episode
        if rec_traj:
            if debugging:
                rewards[i_ep,:], rod_or[i_ep,:], rod_cm[i_ep,:,:], entropies[i_ep,:], values[i_ep,:], target, obstacles, particles, rod,\
                hyp_rod_ang, hyp_perf, perf, perf_rod_ang = \
                    do_episode(agent, parameters, n_step_ep, data_dir, i_ep, rec_traj=rec_traj, train_agent=train_agent, debugging=debugging)
            else:
                rewards[i_ep,:], rod_or[i_ep,:], rod_cm[i_ep,:,:], entropies[i_ep,:], values[i_ep,:], target, obstacles, particles, rod = \
                    do_episode(agent, parameters, n_step_ep, data_dir, i_ep, rec_traj=rec_traj, train_agent=train_agent, debugging=debugging)

            rod_name = f'traj{i_ep}/rod' # name of the dataset in the h5 file has to change for the trajectories
            part_name = f'traj{i_ep}/particles'

            store_file.create_dataset(part_name, compression='gzip', data=particles)
            store_file.create_dataset(rod_name, compression='gzip', data=rod)

            # This is for looking at the hypothetical rods and performances, etc.
            if debugging:

                hyp_rods_name = f'traj{i_ep}/hypRods'          # hypothetical rods
                hyp_perf_name = f'traj{i_ep}/hypPers'          # hypothetical performances
                perfs_name = f'traj{i_ep}/perf'                # performance
                perf_rods_name = f'traj{i_ep}/perfRod'         # rod, from which the performance was determined

                store_file.create_dataset(hyp_rods_name, compression='gzip', data=hyp_rod_ang)
                store_file.create_dataset(hyp_perf_name, compression='gzip', data=hyp_perf)
                store_file.create_dataset(perfs_name, compression='gzip', data=perf)
                store_file.create_dataset(perf_rods_name, compression='gzip', data=perf_rod_ang)

        else:
            rewards[i_ep,:], rod_or[i_ep,:], rod_cm[i_ep,:,:], entropies[i_ep,:], values[i_ep,:], target, obstacles = \
                do_episode(agent, parameters, n_step_ep, data_dir, i_ep, rec_traj=rec_traj, train_agent=train_agent, debugging=debugging)

        # In the case of the transportation problem, the target is saved
        if parameters['mode'] == 7:
                tar_name = f'traj{i_ep}/target'
                store_file.create_dataset(tar_name, compression='gzip', data=target)

        # When obstacles are used, their positions are saved
        if parameters['use_obst']:
                obst_name = f'traj{i_ep}/obstacles'
                store_file.create_dataset(obst_name, compression='gzip', data=obstacles)

        # Get the end time
        ep_end_time = time.time()

        # Print the progress
        print(f"Episode {i_ep} of {n_episodes} in {name} done, took {ep_end_time-ep_start_time} seconds")

        # Save the timing
        elapsed_times[i_ep,0] = ep_end_time - ep_start_time

    store_file.close()


def do_episode_batch_episodic(agent, parameters, data_dir, name, n_episodes, _, *, rec_traj=False, train_agent=False, debugging=False):
    '''
    In the episodic case there is no fixed episode length
    but the episode stops with p = 1-gamma every step (same
    as poisson distributed length with).
    Therefore I need a different setup of the storefile
    '''

    # n_step_ep is the number of simulation steps (observables -> actions -> evolved environment) in one episode
    # n_episodes is the number of episodes to be conducted in this batch

    # Decide on the lengths of the episodes (poisson distribution of lengths)
    poiss_distr_lengths = np.random.poisson(lam=np.round(1/(1-parameters["gamma"])), size=n_episodes)

    # Set up the data storage file in h5 format
    store_file = h5py.File(os.path.join(data_dir, name + '.h5'), 'w')

    for i_ep in range(0, n_episodes):

        # Get the start time
        ep_start_time = time.time()

        # Select the right episode length
        n_step_ep = poiss_distr_lengths[i_ep]

        # Ramp up the trans_dist if required (but not in the evaluation)
        if train_agent and parameters['trans_dist_ramp']:
            parameters['trans_dist'] = min(10 + i_ep * (parameters['max_trans_dist'] - 10)/(n_episodes - 10), parameters['max_trans_dist'])

        if rec_traj:
            if debugging:
                # This is for looking at the hypothetical rods and performances, etc.
                # Execute the episode
                rewards, rod_or, rod_cm, entropies, values, target, particles, \
                rod, hyp_rod_ang, hyp_perf, perf, perf_rod_ang = \
                    do_episode(
                        agent, parameters, n_step_ep, data_dir, i_ep,
                        rec_traj=rec_traj, train_agent=train_agent, debugging=debugging)

                # Things that are saved only if rec_traj and debugging
                store_file.create_dataset(f'traj{i_ep}/hypRods', compression='gzip', data=hyp_rod_ang) # hypothetical rods
                store_file.create_dataset(f'traj{i_ep}/hypPers', compression='gzip', data=hyp_perf) # hypothetical performances
                store_file.create_dataset(f'traj{i_ep}/perf', compression='gzip', data=perf) # performance
                store_file.create_dataset(f'traj{i_ep}/perfRod', compression='gzip', data=perf_rod_ang) # rod, from which the performance was determined
            else:
                # Execute the episode
                rewards, rod_or, rod_cm, entropies, values, target, obstacles, particles, rod = \
                    do_episode(
                        agent, parameters, n_step_ep, data_dir, i_ep,
                        rec_traj=rec_traj, train_agent=train_agent, debugging=debugging)

            # Things that are saved only if rec_traj
            store_file.create_dataset(f'traj{i_ep}/particles', compression='gzip', data=particles)
            store_file.create_dataset(f'traj{i_ep}/rod', compression='gzip', data=rod)

        else:
            # Execute the episode
            rewards, rod_or, rod_cm, entropies, values, target, obstacles = \
                do_episode(
                    agent, parameters, n_step_ep, data_dir, i_ep,
                    rec_traj=rec_traj, train_agent=train_agent, debugging=debugging)

        # Things that are saved in every episode whatsoever
        store_file.create_dataset(f'traj{i_ep}/rewards', compression='gzip', data=rewards)
        store_file.create_dataset(f'traj{i_ep}/rod_or', compression='gzip', data=rod_or)
        store_file.create_dataset(f'traj{i_ep}/rod_cm', compression='gzip', data=rod_cm)
        store_file.create_dataset(f'traj{i_ep}/entropies', compression='gzip', data=entropies)
        store_file.create_dataset(f'traj{i_ep}/values', compression='gzip', data=values)

        # In the case of the transportation problem, the target is saved
        if parameters['mode'] == 7:
                tar_name = f'traj{i_ep}/target'
                store_file.create_dataset(tar_name, compression='gzip', data=target)

        # When obstacles are used, their positions are saved
        if parameters['use_obst']:
                obst_name = f'traj{i_ep}/obstacles'
                store_file.create_dataset(obst_name, compression='gzip', data=obstacles)

        # Get the end time
        ep_end_time = time.time()

        # Print the progress
        print(f"Episode {i_ep} of {n_episodes} in {name} done, took {ep_end_time-ep_start_time} seconds")

    store_file.close()


def do_episode(agent, parameters, n_step_ep, data_dir, i_ep, *, rec_traj=False, train_agent=False, debugging=False):

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

        if debugging:
            # for debugging the hypothetical performance, hyp. rods, performace and the rod, from which
            # the performance was determined are saved, too
            hyp_rod_ang = np.zeros((parameters['N'], n_step_ep), dtype='f4') # just the angle is saved (self.N x n_step_ep values)
            hyp_perf = np.zeros((parameters['N'], n_step_ep), dtype='f4') # (self.N x n_step_ep values)
            perf = np.zeros((n_step_ep), dtype='f4') # the overall performance in one timestep
            perf_rod_ang = np.zeros((n_step_ep), dtype='f4') # the rod, from which perf was determined

    # Initialize the environment
    environment = MD_ROD(**parameters)
    obs, rewards = environment.get_obs_rewards() # gets first obs and rewards

    # Initialize the agent
    agent.initialize(obs)

    # Real simulation loop
    for step in range(n_step_ep):

        # Agent decides actions from the observables
        actions, logp = agent.get_actions()

        # The environment is updated according to the selected actions
        obs, rewards, rod_theta, rod_com = environment.evolve_MD(actions)

        # Determine if the episode should end
        final = parameters["episodic"] and (environment.task_achieved or step == n_step_ep)

        # Add the environment response to the knowledge od the agent
        values = agent.add_environment_response(environment.lost, obs, rewards, final=final)

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

            if debugging:
                # for debugging the hypothetical performance, hyp. rods, performace and the rod, from which
                # the performance was determined are saved, too
                hyp_rod_ang[:,step] = environment.hyp_rod_ang
                hyp_perf[:,step] = environment.hyp_perf
                perf[step] = environment.performance
                perf_rod_ang[step] = environment.perf_rod_ang

        # Train the agent
        if train_agent and ((step+1) % parameters['train_pause'] == 0 or final):
            try:
                agent.train_step()
            except ValueError as e:
                print("Error in train step:", e)
                print("Observables", agent.observables)
                print("Estimated Return", agent.estimated_return)
                print("Step:", step)
                print("Episode:", i_ep)

            agent.initialize(obs)

            # Save checkpoints of both actor and critic for evaluation
            agent.save_weights(os.path.join(data_dir, 'model'), step+1 + i_ep * n_step_ep)

        # In the case of truly episodic tasks, check whether the aim is achieved
        # and if so, end the episode early
        if final:
            break

    agent.finish_episode()
    if not train_agent:
        agent.reset_memory()

    if rec_traj:
        if debugging:
            return mean_rew, rod_or, rod_cm, mean_ent, mean_val, environment.target, environment.obstacles, particles, rod, hyp_rod_ang, hyp_perf, perf, perf_rod_ang
        else:
            return mean_rew, rod_or, rod_cm, mean_ent, mean_val, environment.target, environment.obstacles, particles, rod
    else:
        return mean_rew, rod_or, rod_cm, mean_ent, mean_val, environment.target, environment.obstacles


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
