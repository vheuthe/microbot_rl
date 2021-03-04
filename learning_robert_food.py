import numpy as np
import sys
import os
import json

from firstrl import AgentActiveMatter
from md_env_fortran import MD



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
    'training_frequency': 360,
    'training_epochs': 50,
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
    'cones': 5,
    'cone_angle': 180,
    'vel_act': 0.5, # 0.35
    'vel_tor': 0.35, # 0.2
    'torque': 25,
}



def do_task(parameters, data_dir):

    # initialize data folder
    os.makedirs(data_dir, exist_ok=True)

    # create and save full parameter set
    parameters = default_parameters.copy()
    parameters.update(parameters)
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
    
    # set up environment
    md = MD(
        md_type = 'food',
        index = run_id,
        size = 100, # unused?
        steps = parameters['action_time']/parameters['dt'],
        traj = True, # for now
        data_dir = data_dir,
        **parameters
    )

    # start with closeby food
    food_x = food_y = parameters['food_width']/3
    food_amount = parameters['food_amount']
    food_width = parameters['food_width']
    food_wait = parameters['food_delay']

    # initialize Agent
    obs, rew, eaten = md.get_obs_rewards_food(food_x, food_y, food_amount, food_width)
    agent.initialize(obs)

    for step in range(int(parameters['total_time']/parameters['action_time'])):

        print(run_id, step)

        # get actions
        actions, logp = agent.get_actions(flag_logp=True)

        # renew food if necessary
        if food_amount <= 0:
            food_width = 0
            food_wait -= 1
            if food_wait <= 0:
                theta = np.random.rand()*np.pi*2
                displ = np.random.normal(parameters['food_dist'], parameters['food_dist']/2)
                food_x += displ * np.cos(theta)
                food_y += displ * np.sin(theta)
                food_amount = parameters['food_amount']
                food_width = parameters['food_width']
                food_wait = parameters['food_delay']

        # evolve brownian dynamics (fortran)
        obs, rew, eaten, _done, _info = md.evolve_MD(actions.astype(int), XP=food_x, YP=food_y, Food=food_amount, Food_width=food_width)
        food_amount -= eaten

        # save to file
        md.print_xyz_food_actions(food_x, food_y, food_amount, food_width, logp, actions.astype(int))
        
        # add rewrads
        agent.add_env_timeframe([], obs, rew)

        # train model
        if (step + 1) % parameters['training_frequency'] == 0:
            agent.train_step(epochs=parameters['training_epochs'])
            agent.initialize(obs)
    
    # clean up unfinished trajectories
    agent.finish_episode()



if __name__ == "__main__":
    # mainly for testing
    if len(sys.argv) > 1:
        do_task({}, sys.argv[1])
    else:
        do_task({}, 'sim-test')
