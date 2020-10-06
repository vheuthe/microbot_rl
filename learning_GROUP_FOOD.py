# TEST WITH MD
import numpy as np
import sys
from md_env_fortran import MD
from firstrl import AgentActiveMatter
import scipy
import time

# FOR MD

# ------------
n_input = 20
# ------------

n_actions = 4
gam = 0.85
lam = 0.97
CL =0.15
en_coeff=0.02

restart=False
models_rootname = 'models_GroupFood'

Agent = AgentActiveMatter(input_dim=n_input, output_dim=n_actions, en_coeff=en_coeff, CL=CL, gamma=gam, models_rootname=models_rootname, lam=lam, lrP=0.0003, lrV=0.001,  restart_models=restart)

# ------------------------------------------------

start_MD = 0
n_MD = 1

total_time = 3600
step_time = 5
vel_act = 0.35
vel_tor = 0.20
dt = 0.1
steps = int(step_time/dt)
n_max_steps = int(total_time/step_time)
steps_update = 128

torque = 25 #875. / 180. # torque in rad

N = 20 #number of particles

# ============
count = 0
# ============

for iMD in range(start_MD, start_MD + n_MD):
        # reinitialize the class MD with the new index

        theta = np.random.rand()*2*np.pi
        P = 80*np.array([np.cos(theta),np.sin(theta),0.])
        traj_flag = False
        if (iMD%5 == 0):
            traj_flag=True
        md = MD(md_type='food', index=iMD, N=N, size=100, steps=steps, vel_act=vel_act, vel_tor=vel_tor, food_rew=15, dt=dt, torque=torque, traj=traj_flag, cones=10, cone_angle=360)
        traj_file = open('traj'+str(iMD)+'.xyz', 'a')
        print('\n\n\n #NEW MD INITIALIZATION!')

        obs, rewards = md.get_obs_rewards_food(XP=P[0], YP=P[1]) # gets first obs and advantages
        Agent.initialize(obs)

        # -----------------------------------------
        for step in range(n_max_steps):
            count += 1
            actions = Agent.get_actions() #return actions vector to give particles, and label
            if (step == n_max_steps//2):
                theta += (np.random.randint(2)+1)*np.pi/3.*2
                P = 80*np.array([np.cos(theta),np.sin(theta),0.])
            obs, rewards, done, info = md.evolve_MD(actions.astype(int), XP=P[0], YP=P[1]) #evolve systems from given actions
            #print(obs)
            md.print_xyz_food(P[0], P[1])
            if ((step>0) and (step%steps_update == 0)):
                lost = [i for i in range(obs.shape[0])]
                Agent.add_env_timeframe(lost, obs, rewards, done)
                Agent.train_step(epochs=50)
                Agent.initialize(obs)
            else:
                Agent.add_env_timeframe([], obs, rewards, False)
            print('{} {} {} {} {} {}'.format(iMD, step, np.sum(rewards), P[0], P[1], P[2]))

Agent.save_models(path=models_rootname+'gamma0p85', final_save = True)

