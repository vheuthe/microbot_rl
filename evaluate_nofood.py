import numpy as np
import scipy.stats
import scipy.io
import sys
import os
import json

from firstrl import AgentActiveMatter
from environments.food import FoodEnvironment


assert len(sys.argv) > 1

root_dir = os.path.abspath(sys.argv[1])

assert os.path.isdir(root_dir)

tasks = [entry for entry in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, entry))]

for task in tasks:

    with open(os.path.join(root_dir, task, 'parameters.json')) as reader:
        parameters = json.load(reader)

    agent = AgentActiveMatter(
        models_rootname = os.path.join(root_dir, task, 'model'), 
        restart_models = True,
        **parameters
    )

    parameters['food_mode'] = 'none'
    environment = FoodEnvironment(**parameters)

    # # data saving
    # stats_file = open(os.path.join(root_dir, task, 'nofood_stats.xyz'), 'w')
    # traj_file = open(os.path.join(root_dir, task, 'nofood_traj.xyz'), 'w')

    # initialize
    observables = environment.reset(parameters['N'])
    agent.initialize(observables)

    # create "cell-arrays" for saving
    max_step = int(parameters['total_time']/parameters['action_time'])
    data = {
        'food': np.empty((1,max_step), dtype=object),
        'particles': np.empty((1,max_step), dtype=object),
        'stats': np.empty((1,max_step), dtype=object),
        'actions': np.empty((1,max_step), dtype=object),
        'observables': np.empty((1,max_step), dtype=object),
    }

    for step in range(max_step):

        print(task, step)

        # get actions
        actions, logp = agent.get_actions(flag_logp=True)

        # adjust actions if there is no passive one
        if agent.n_actions == 3:
            actions += 1

        # get environment response
        observables, rewards = environment.evolve(actions)
        
        # add environment response
        values = agent.add_env_timeframe([], observables, rewards)

        # Save stats
        entropies = scipy.stats.entropy(np.exp(logp), base=agent.n_actions, axis=1)

        data['food'][0, step] = np.array([])
        data['particles'][0, step] = environment.particles
        data['stats'][0, step] = np.hstack((rewards.reshape(-1,1), entropies.reshape(-1,1), values.reshape(-1,1)))
        data['actions'][0, step] = np.hstack((actions.reshape((-1,1)), logp))
        data['observables'][0, step] = observables


        # stats_file.write('{} {} {} {}\n'.format(step, np.mean(rewards), np.mean(entropies), np.mean(values)))
        # # stick to emanueles format for now
        # traj_file.write('\n\n')
        # for f in environment.food:
        #     traj_file.write('1 {} {} 0 0 0 {} {} 0\n'.format(*f[0:4]))
        # for p, r, a in zip(environment.particles, rewards, actions):
        #     traj_file.write('0 {} {} 0 {} {} {} 6.2 {}\n'.format(p[0], p[1], np.cos(p[2]), np.sin(p[2]), r, a))

        # no training!

    scipy.io.savemat(os.path.join(root_dir, task, 'nofood.mat'), data)

    # # clean up io
    # stats_file.close()
    # traj_file.close()