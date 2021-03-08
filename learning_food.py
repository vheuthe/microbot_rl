import numpy as np
import scipy.stats
import sys
import os
import json

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
    'model_structure': [(32, 'relu'),(16, 'relu'),(16, 'relu')],

    # Learning
    'food_rew': 0.6,
    'touch_penalty': 3,
    'obs_type': '1overR',
    'cones': 5,
    'cone_angle': 180,
    'training_frequency': 360,
    'training_epochs': 50,
    'food_mode': 'random',
    'food_dist': 200, # distance for new food
    'food_amount': 2000,
    'food_width': 150,
    'food_delay': 100,

    # Runs
    'N': 30,
    'n_start': 0,
    'n_stop': 100,
    'dt': 0.2,
    'action_time': 6,
    'total_time': 7200,
    'vel_act': 0.5,
    'sig_vel_act': 0.25,
    'vel_tor': 0.35,
    'sig_vel_tor': 1.75,
    'torque': 25,
}



def do_task(selected_parameters, data_dir):

    # initialize data folder
    os.makedirs(data_dir, exist_ok=True)

    # create and save full parameter set
    parameters = default_parameters.copy()
    parameters.update(selected_parameters)
    with open(os.path.join(data_dir, 'parameters.json'), 'w', encoding='utf-8') as paramfile:
        json.dump(parameters, paramfile, ensure_ascii=False, indent=4)

    # instantiate agent with new neural networks 
    agent = AgentActiveMatter(
        models_rootname = os.path.join(data_dir, 'model'), 
        restart_models = False,
        **parameters
    )

    # loop over simulations, sequentially improving the same network
    for run_id in range(parameters['n_start'], parameters['n_stop']):
        do_run(run_id, agent, data_dir, parameters)
        if (run_id + 1) % 20 == 0:
            agent.save_models(os.path.join(data_dir, 'model'), final_save = True)
    
    agent.save_models(os.path.join(data_dir, 'model'), final_save = True)



def do_run(run_id, agent, data_dir, parameters):
    
    # set up
    environment = FoodEnvironment(**parameters)

    # data saving
    stats_file = open('{}/stats{:02d}.xyz'.format(data_dir, run_id), 'w')
    if run_id % 10 == 9:
        traj_file = open('{}/traj{:02d}.xyz'.format(data_dir, run_id), 'w')

    # initialize
    observables = environment.reset(parameters['N'])
    agent.initialize(observables)

    for step in range(int(parameters['total_time']/parameters['action_time'])):

        print(run_id, step)

        # get actions
        actions, logp = agent.get_actions(flag_logp=True)

        # adjust actions if there is no passive one
        if agent.n_actions == 3:
            actions += 1

        # get environment response
        observables, rewards = environment.evolve(actions)

        # save to file
        #md.print_xyz_food_actions(food_x, food_y, food_amount, food_width, logp, actions.astype(int))
        
        # add environment response
        values = agent.add_env_timeframe([], observables, rewards)

        # Save stats
        entropies = scipy.stats.entropy(np.exp(logp), base=agent.n_actions, axis=1)

        stats_file.write('{} {} {} {}\n'.format(step, np.mean(rewards), np.mean(entropies), np.mean(values)))
        if run_id % 10 == 9:
            # stick to emanueles format for now
            traj_file.write('\n\n')
            for f in environment.food:
                traj_file.write('1 {} {} 0 0 0 {} {} 0\n'.format(*f[0:4]))
            for p, r, a in zip(environment.particles, rewards, actions):
                traj_file.write('0 {} {} 0 {} {} {} 6.2 {}\n'.format(p[0], p[1], np.cos(p[2]), np.sin(p[2]), r, a))

        # train model
        if (step + 1) % parameters['training_frequency'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(observables)
    
    # clean up unfinished trajectories
    agent.finish_episode()

    # clean up io
    stats_file.close()
    if run_id % 10 == 9:
        traj_file.close()



if __name__ == "__main__":
    # mainly for testing
    if len(sys.argv) > 1:
        do_task({}, sys.argv[1])
    else:
        do_task({}, 'sim-test')
