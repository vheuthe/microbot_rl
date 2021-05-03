import itertools
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

    # sequentially train the agent,
    # write trajectories only every 10th episode
    for episode in range(100):
        do_episode(
            agent,
            parameters,
            stats_file = '{}/train_{:02d}_stats.xyz'.format(data_dir, episode),
            traj_file = None if (episode + 1) % 10 else '{}/train_{:02d}_traj.xyz'.format(data_dir, episode),
            train_agent = True,
            stop_time = 2*3600,
        )
        agent.save_weights(os.path.join(data_dir, 'model'), 'train_{:02d}'.format(episode))

    # increase episode length
    for episode in range(100,150):
        do_episode(
            agent,
            parameters,
            stats_file = '{}/train_{:02d}_stats.xyz'.format(data_dir, episode),
            traj_file = None if (episode + 1) % 10 > 0 else '{}/train_{:02d}_traj.xyz'.format(data_dir, episode),
            train_agent = True,
            stop_time = 10*3600,
        )
        agent.save_weights(os.path.join(data_dir, 'model'), 'train_{:02d}'.format(episode))

    # no training after this point
    agent.save_models(os.path.join(data_dir, 'model'))

    # do some episodes with fixed seeds for evaluation
    for seed in range(5):
        do_episode(
            agent,
            parameters,
            stats_file = '{}/evaluate_{:02d}_stats.xyz'.format(data_dir, seed),
            traj_file = None if seed > 0 else '{}/evaluate_{:02d}_traj.xyz'.format(data_dir, seed),
            seed = seed,
            # stop_food_counter = 11,
            stop_time = 10*3600, # as backup
        )

    # do one episode without food to evaluate steady state behavior
    do_episode(
        agent,
        {**parameters, 'food_mode': 'none'},
        stats_file = '{}/nofood_stats.xyz'.format(data_dir),
        traj_file = '{}/nofood_traj.xyz'.format(data_dir),
        seed = 0,
        stop_time = 10*3600,
    )



def do_episode(agent, parameters, *, stats_file=None, traj_file=None, train_agent=False, seed=None, stop_time=np.inf, stop_food_counter=np.inf):

    # argument checks
    assert np.isfinite(stop_time) or np.isfinite(stop_food_counter), "No stop condition set!"

    if stats_file:
        stats_file = open(stats_file, 'w')
    if traj_file:
        traj_file = open(traj_file, 'w')

    environment = FoodEnvironment(**parameters)
    observables = environment.reset(parameters['N'], seed=seed)
    agent.initialize(observables)

    for step in itertools.count():

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
        entropies = scipy.stats.entropy(np.exp(logp), base=agent.n_actions, axis=1)
        if stats_file:
            stats_file.write('{} {} {} {}\n'.format(step, np.mean(rewards), np.mean(entropies), np.mean(values)))
        if traj_file:
            # stick to emanueles format for now
            traj_file.write('\n\n')
            for f in environment.food:
                traj_file.write('1 {} {} 0 0 0 {} {} 0\n'.format(*f[0:4]))
            for p, r, a in zip(environment.particles, rewards, actions):
                traj_file.write('0 {} {} 0 {} {} {} 6.2 {}\n'.format(p[0], p[1], np.cos(p[2]), np.sin(p[2]), r, a))

        # train model
        if train_agent and (step + 1) % parameters['training_frequency'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(observables)

        # stop episode
        if step*parameters['action_time'] >= stop_time or environment.food_counter >= stop_food_counter:
            break

    # clean up unfinished trajectories
    agent.finish_episode()
    if not train_agent:
        agent.reset_memory()

    if stats_file:
        stats_file.close()
    if traj_file:
        traj_file.close()



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
    if len(sys.argv) > 1:
        do_task({}, sys.argv[1])
    else:
        do_task({}, 'sim-test')
