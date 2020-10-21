# TEST WITH MD
import numpy as np
import sys
from md_env_fortran import MD
from firstrl import AgentActiveMatter
import scipy
import time

# FOR MD

# ------------
n_input = 15
# ------------

n_actions = 4
gam = 0.95
lam = 0.97
CL = 0.03
en_coeff = 0.01

# ------------------------------------------------

start_MD = 0
n_MD = 100

total_time = 3600
step_time = 5
vel_act = 0.35
vel_tor = 0.20
dt = 0.15
steps = int(step_time/dt)
n_max_steps = int(total_time/step_time)
steps_update = 720
starting_food = 1000
eating_speed = 1

torque = 25 #875. / 180. # torque in rad
N = 30 #number of particles

food_rew = 0.5  # 1=only food, 0=only compactness
# ---------------------------------

restart = False
if (start_MD > 0):
    restart = True
models_rootname = 'models_GroupFood_dOBS'
model_structure = [(32, 'relu'),(16, 'relu'),(16, 'relu')]

Agent = AgentActiveMatter(input_dim=n_input, output_dim=n_actions, en_coeff=en_coeff, CL=CL, gamma=gam, models_rootname=models_rootname, lam=lam, lrP=0.0003, lrV=0.001,  restart_models=restart, model_structure = model_structure)

# ============
# FUCKING HELL
count = 0
# ============


for iMD in range(start_MD, start_MD + n_MD):
        # reinitialize the class MD with the new index

        theta = np.random.rand()*2*np.pi
        P = 1500*np.array([np.cos(theta),np.sin(theta),0.])
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
            actions = Agent.get_actions() #return actions vector to give particles, and label
            if (Food_quantity < -50):
                theta += np.random.rand()*np.pi*2
                P += 150*np.array([np.cos(theta),np.sin(theta),0.])
                Food_quantity = starting_food
            obs, rewards, Eaten, done, info = md.evolve_MD(actions.astype(int), XP=P[0], YP=P[1], Food=Food_quantity) #evolve systems from given actions
            Food_quantity -= Eaten*eating_speed
 
            md.print_xyz_food(P[0], P[1], Food_quantity)
            if ((step>0) and (step%steps_update == 0)):
                lost = [i for i in range(obs.shape[0])]
                Agent.add_env_timeframe(lost, obs, rewards, done)
                Agent.train_step(epochs=50)
                Agent.initialize(obs)
            else:
                Agent.add_env_timeframe([], obs, rewards, False)
            print('{} {} {} {} {} {}'.format(iMD, step, np.sum(rewards), P[0], P[1], Food_quantity))

Agent.save_models(path=models_rootname, final_save = True)

