import numpy as np
import scipy.stats
import sys
import os
import json

from firstrl import AgentActiveMatter
from environments.food import FoodEnvironment


root_dir = '/data/scc/robert.loeffler/2021-03-10-rl-alternating'

for task in os.listdir(root_dir):

    with open(os.path.join(root_dir, task, 'parameters.json')) as reader:
        parameters = json.load(reader)

    agent = AgentActiveMatter(
        models_rootname = os.path.join(root_dir, task, 'model'), 
        restart_models = True,
        **parameters
    )

    parameters['food_mode'] = 'none'
    environment = FoodEnvironment(**parameters)

    # data saving
    stats_file = open(os.path.join(root_dir, task, 'nofood_stats.xyz'), 'w')
    traj_file = open(os.path.join(root_dir, task, 'nofood_traj.xyz'), 'w')

    # initialize
    observables = environment.reset(parameters['N'])
    agent.initialize(observables)

    for step in range(int(parameters['total_time']/parameters['action_time'])):

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

        stats_file.write('{} {} {} {}\n'.format(step, np.mean(rewards), np.mean(entropies), np.mean(values)))
        # stick to emanueles format for now
        traj_file.write('\n\n')
        for f in environment.food:
            traj_file.write('1 {} {} 0 0 0 {} {} 0\n'.format(*f[0:4]))
        for p, r, a in zip(environment.particles, rewards, actions):
            traj_file.write('0 {} {} 0 {} {} {} 6.2 {}\n'.format(p[0], p[1], np.cos(p[2]), np.sin(p[2]), r, a))

        # no training!

    # clean up io
    stats_file.close()
    traj_file.close()