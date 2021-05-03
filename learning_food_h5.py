import h5py
import json
import numpy as np
import scipy.stats
import sys
import os

from firstrl import AgentActiveMatter
from environments.food import FoodEnvironment


default_parameters = {

    # RL Agent
    'input_dim': 20,
    'output_dim': 4,
    'en_coeff': 0.0,
    'CL': 0.07,
    'gamma': 0.97,
    'lam': 0.97,
    'lrPI': 0.003,
    'lrV': 0.003,
    'target_kl': 0.02,
    'model_structure': [(32, 'relu'),(16, 'relu'),(16, 'relu')],

    # Training
    'food_rew': 0.6,
    'touch_penalty': 0, # 3,
    'tp_type': 'all',
    'max_nn_rew': 999,
    'obs_type': '1overR',
    'cones': 5,
    'rew_cones': 2,
    'vision_angle': 180,
    'visual_particle_size': 6.2,
    'training_frequency': 240,
    'training_epochs': 50,
    'food_mode': 'randombox',
    'food_dist': 200, # distance for new food
    'food_amount': 2000,
    'food_width': 100,
    'food_delay': 100,

    # Episodes
    'N': 30,
    'particle_size': 6.2,
    'training_mode': 'dynamic',
    'dt': 0.2,
    'action_time': 6,
    'Dt': 0, # 0.014,
    'Dr': 0, # 1.0 / 350.0,
    'vel_act': 0.5,
    'vel_tor': 0.35,
    'vel_noise': 0, # 0.5, relative!
    'torque': 25,
}


def do_array_task(task_id, job_dir):

    # parameter ranges are stored in the job_dir
    with open(os.path.join(job_dir, 'parameters.json'), 'r') as reader:
        job_parameters = json.load(reader)

    # choose one set out of all possible parameter combinations
    # (task_id's start at 1 !!)
    selected_parameters = dict(zip(
        job_parameters.keys(),
        [vals.flat[task_id - 1] for vals in np.meshgrid(*job_parameters.values())]
    ))

    # construct folder name from relevant parameters
    data_dir = os.path.join(
        job_dir,
        '_'.join([key + str(val) for key, val in selected_parameters.items()])
    )

    do_task(selected_parameters, data_dir)


def do_task(selected_parameters, data_dir):

    # initialize data folder
    os.makedirs(data_dir, exist_ok=True)

    # create and save full parameter set
    parameters = default_parameters.copy()
    parameters.update(selected_parameters)
    with open(os.path.join(data_dir, 'parameters.json'), 'w', encoding='utf-8') as paramfile:
        json.dump(parameters, paramfile, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # instantiate agent with new neural networks
    agent = AgentActiveMatter(
        models_rootname = os.path.join(data_dir, 'model'),
        restart_models = False,
        **parameters
    )
    agent.save_models(os.path.join(data_dir, 'model'))

    # setup reusable environment
    environment = FoodEnvironment(**parameters)

    # - - - - - - - - - -
    # sequentially train the agent

    steps = int(2*3600 / parameters['action_time'])
    episodes = 100

    storage = h5py.File(os.path.join(data_dir, 'train_short.hdf5'), 'w-')
    rewards = storage.create_dataset('/rewards', (episodes,steps), dtype='f4', compression='gzip')
    entropies = storage.create_dataset('/entropies', (episodes,steps), dtype='f4', compression='gzip')
    values = storage.create_dataset('/values', (episodes,steps), dtype='f4', compression='gzip')

    for i in range(episodes):
        rewards[i,:], entropies[i,:], values[i,:] = do_episode(agent, environment, parameters, steps, train_agent=True)
        agent.save_weights(os.path.join(data_dir, 'model'), 'train_short_{:02d}'.format(i))

    storage.close()

    # - - - - - - - - - -
    # increase episode length

    steps = int(10*3600 / parameters['action_time'])
    episodes = 50

    storage = h5py.File(os.path.join(data_dir, 'train_long.hdf5'), 'w-')
    rewards = storage.create_dataset('/rewards', (episodes,steps), dtype='f4', compression='gzip')
    entropies = storage.create_dataset('/entropies', (episodes,steps), dtype='f4', compression='gzip')
    values = storage.create_dataset('/values', (episodes,steps), dtype='f4', compression='gzip')

    for i in range(episodes):
        rewards[i,:], entropies[i,:], values[i,:] = do_episode(agent, environment, parameters, steps, train_agent=True)
        agent.save_weights(os.path.join(data_dir, 'model'), 'train_long_{:02d}'.format(i))

    storage.close()

    # no training after this point
    agent.save_models(os.path.join(data_dir, 'model'))

    # - - - - - - - - - -
    # do some episodes with fixed seeds for evaluation

    steps = int(10*3600 / parameters['action_time'])
    seeds = 5

    storage = h5py.File(os.path.join(data_dir, 'evaluate.hdf5'), 'w-')
    rewards = storage.create_dataset('/rewards', (seeds,steps), dtype='f4', compression='gzip')
    entropies = storage.create_dataset('/entropies', (seeds,steps), dtype='f4', compression='gzip')
    values = storage.create_dataset('/values', (seeds,steps), dtype='f4', compression='gzip')

    rewards[i,:], entropies[i,:], values[i,:], food, particles = do_episode(agent, environment, parameters, steps, seed=0, record_traj=True)
    storage.create_dataset('/traj_00/food', compression='gzip', data=food)
    storage.create_dataset('/traj_00/particles', compression='gzip', data=particles)

    for i in range(1, seeds):
        rewards[i,:], entropies[i,:], values[i,:] = do_episode(agent, environment, parameters, steps, seed=i)

    storage.close()

    # - - - - - - - - - -
    # do one episode without food to evaluate steady state behavior

    nofood_environment = FoodEnvironment(**{**parameters, 'food_mode': 'none'})
    steps = int(10*3600 / parameters['action_time'])
    storage = h5py.File(os.path.join(data_dir, 'nofood.hdf5'), 'w-')

    rewards, entropies, values, food, particles = do_episode(agent, nofood_environment, parameters, steps, seed=0, record_traj=True)

    storage.create_dataset('/rewards', compression='gzip', data=rewards.reshape((1,-1)))
    storage.create_dataset('/entropies', compression='gzip', data=entropies.reshape((1,-1)))
    storage.create_dataset('/values', compression='gzip', data=values.reshape((1,-1)))
    storage.create_dataset('/traj_00/food', compression='gzip', data=food)
    storage.create_dataset('/traj_00/particles', compression='gzip', data=particles)



def do_episode(agent, environment, parameters, steps, *, record_traj=False, train_agent=False, seed=None):

    #steps = int(episode_length / parameters['action_time'])

    # save stats as single precision floats (32bit == 4byte)
    mean_reward = np.zeros((steps), dtype='f4')
    mean_entropy = np.zeros((steps), dtype='f4')
    mean_value = np.zeros((steps), dtype='f4')

    observables = environment.reset(parameters['N'], seed=seed)
    agent.initialize(observables)

    if record_traj:
        food = np.full((*environment.food.shape, steps+1), np.nan)
        particles = np.full((parameters['N'], 5, steps+1), np.nan)
        food[:,:,0] = environment.food
        particles[:,0:3,0] = environment.particles

    # main loop
    for step in range(0, steps):

        # get actions
        actions, logp = agent.get_actions()

        # adjust actions if there is no passive one
        if agent.n_actions == 3:
            actions += 1

        # get environment response
        observables, rewards = environment.evolve(actions)

        # add environment response
        values = agent.add_environment_response([], observables, rewards)

        # Save stats
        mean_reward[step] = np.mean(rewards)
        mean_entropy[step] = np.mean(scipy.stats.entropy(np.exp(logp), base=agent.n_actions, axis=1))
        mean_value[step] = np.mean(values)

        if record_traj:
            food[:,:,step+1] = environment.food
            particles[:,4,step] = actions
            particles[:,0:3,step+1] = environment.particles
            particles[:,3,step+1] = rewards

        # train model
        if train_agent and (step + 1) % parameters['training_frequency'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(observables)


    # clean up unfinished trajectories
    agent.finish_episode()
    if not train_agent:
        agent.reset_memory()

    if not record_traj:
        return mean_reward, mean_entropy, mean_value
    else:
        return mean_reward, mean_entropy, mean_value, food, particles



class NumpyEncoder(json.JSONEncoder):
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



if __name__ == "__main__":
    # mainly for testing

    if len(sys.argv) > 2:
        task_id = int(sys.argv[2])
        job_dir = os.path.abspath(sys.argv[1])
        do_array_task(task_id, job_dir)

    elif len(sys.argv) > 1:
        do_task({}, sys.argv[1])

    else:
        do_task({}, 'sim-test')
