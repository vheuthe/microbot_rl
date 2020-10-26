# TEST WITH MD
import numpy as np
import sys
from md_env_fortran import MD
from firstrl import AgentActiveMatter
import scipy
import time

if __name__ == "__main__":
    # READS FOOD_REW AS INPUT
    food_rew_input = np.float(sys.argv[1])

    # ------------
    n_input = 15
    # ------------

    n_actions = 3
    gam = 0.90
    lam = 0.97
    CL = 0.03
    en_coeff = 0.01

    # ------------------------------------------------

    start_MD = 0
    n_MD = 200

    total_time = 3600
    step_time = 10
    vel_act = 0.35
    vel_tor = 0.20
    dt = 0.15
    steps = int(step_time/dt)
    n_max_steps = int(total_time/step_time)
    steps_update = 128
    starting_food = 2000
    eating_speed = 3
    displ = 150

    torque = 25 #875. / 180. # torque in rad
    N = 50 #number of particles

    food_rew = food_rew_input  # 1=only food, 0=only compactness
    # ---------------------------------

    restart = False
    if (start_MD > 0):
        restart = True
    models_rootname = 'models_GroupFood_dOBS_FoodRew{}_EatSpeed{}'.format(food_rew, eating_speed)
    model_structure = [(32, 'relu'),(32,'relu'),(16, 'relu'),(16, 'relu')]

    Agent = AgentActiveMatter(input_dim=n_input, output_dim=n_actions, en_coeff=en_coeff, CL=CL, gamma=gam, models_rootname=models_rootname, lam=lam, lrP=0.001, lrV=0.001,  restart_models=restart, model_structure = model_structure)

    # ============
    count = 0
    # ============


    for iMD in range(start_MD, start_MD + n_MD):
            # reinitialize the class MD with the new index

            theta = np.random.rand()*2*np.pi
            P = displ*np.array([np.cos(theta),np.sin(theta),0.])
            Food_quantity = starting_food
            traj_flag = False
            if (iMD%1 == 0):
                traj_flag=True
            md = MD(md_type='food', index=iMD, N=N, size=100, steps=steps, vel_act=vel_act, vel_tor=vel_tor, food_rew=food_rew, dt=dt, torque=torque, traj=traj_flag, cones=5, cone_angle=180)
            traj_file = open('traj'+str(iMD)+'.xyz', 'a')
            print('\n\n\n #NEW MD INITIALIZATION!')
            
            obs, rewards, Eaten = md.get_obs_rewards_food(XP=P[0], YP=P[1], Food=Food_quantity) # gets first obs and advantages
            Food_quantity -= Eaten*eating_speed
            
            Agent.initialize(obs)
            # -----------------------------------------
            for step in range(n_max_steps):
                count += 1
                actions, logp = Agent.get_actions(flag_logp=True) #return actions vector to give particles, and label
                actions +=1
                
                if ((starting_food > 0) and (Food_quantity < 20)):
                    theta += np.random.rand()*np.pi*2
                    P += displ*np.array([np.cos(theta),np.sin(theta),0.])
                    Food_quantity = starting_food
                obs, rewards, Eaten, done, info = md.evolve_MD(actions.astype(int), XP=P[0], YP=P[1], Food=Food_quantity) #evolve systems from given actions
                Food_quantity -= Eaten*eating_speed
     
                md.print_xyz_food_actions(P[0], P[1], Food_quantity, logp, actions.astype(int))
                if ((step>0) and (step%steps_update == 0)):
                    lost = [i for i in range(obs.shape[0])]
                    Agent.add_env_timeframe(lost, obs, rewards, done)
                    Agent.train_step(epochs=50)
                    Agent.initialize(obs)
                else:
                    Agent.add_env_timeframe([], obs, rewards, False)
                order,  swirl = md.get_order()
                print('{} {} {} {} {} {} {} {}'.format(iMD, step, np.sum(rewards), P[0], P[1], Food_quantity, order, swirl))

    Agent.save_models(path=models_rootname, final_save = True)

