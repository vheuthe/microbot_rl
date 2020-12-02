# TEST WITH MD
import numpy as np
import sys
from md_env_fortran import MD
from firstrl import AgentActiveMatter
import scipy
import time


# FOR KL convergence
def Pi(obs, policy):
    logp = policy(np.array(obs))
    prob = np.exp(logp)
    prob = prob / np.sum(prob,axis=1,keepdims=True)
    return prob

def from_policy_to_actions(Pi):
    '''
    takes distribution of log probabilities over discrete set of actions
    and gives out one randomly, after normalization
    MUST RETURN only one value: index of action!
    '''
    action=np.random.choice(4,p=Pi)
    return action

def KL_symm(A,B):
    return 0.5*np.mean(entropy(A,B,axis=1)+entropy(B,A,axis=1))

if __name__ == "__main__":
    # READS FOOD_REW AS INPUT
    food_rew_input = np.float(sys.argv[1])
    eating_speed = np.float(sys.argv[2])
    touch_penalty = np.float(sys.argv[3])

    # ------------
    cones = 5
    n_input = 3*cones
    cone_angle = 180
    # ------------

    n_actions = 4
    gam = 0.95
    lam = 0.97
    CL = 0.03
    en_coeff = 0.01

    # ------------------------------------------------

    start_MD = 0
    n_MD = 200

    total_time = 7200
    step_time = 5
    vel_act = 0.35
    vel_tor = 0.20
    dt = 0.2
    steps = int(step_time/dt)
    n_max_steps = int(total_time/step_time)
    steps_update = 360
    starting_food = 2000
    starting_food_width = 100
    Food_width = starting_food_width
    torque = 25 #875. / 180. # torque in rad
    N = 50 #number of particles
    food_rew = food_rew_input  # 1=only food, 0=only compactness
    # ---------------------------------

    restart = False
    if (start_MD > 0):
        restart = True
    models_rootname = 'models_GroupFood_dOBS_FoodRew{}_EatSpeed{}'.format(food_rew, eating_speed)
    model_structure = [(32, 'relu'),(16, 'relu'),(16, 'relu')]

    Agent = AgentActiveMatter(input_dim=n_input, output_dim=n_actions, en_coeff=en_coeff, CL=CL, gamma=gam, models_rootname=models_rootname, lam=lam, lrP=0.001, lrV=0.001,  restart_models=restart, model_structure = model_structure)

    # ============
    count = 0
    wait = 100
    # ============

    for iMD in range(start_MD, start_MD + n_MD):
            # reinitialize the class MD with the new index
            displ = 75
            theta = np.random.rand()*2*np.pi
            P = displ*np.array([np.cos(theta),np.sin(theta),0.])
            Food_quantity = starting_food
            Food_width = starting_food_width
            traj_flag = False
            if (iMD%50 == 49):
                traj_flag=True
            md = MD(md_type='food', index=iMD, obs_type='1overR', N=N, size=100, steps=steps, vel_act=vel_act, vel_tor=vel_tor, food_rew=food_rew, touch_penalty=touch_penalty, dt=dt, torque=torque, traj=traj_flag, cones=cones, cone_angle=cone_angle)
            traj_file = open('traj'+str(iMD)+'.xyz', 'a')
            print('\n\n\n #NEW MD INITIALIZATION!')
            
            obs, rewards, Eaten = md.get_obs_rewards_food(XP=P[0], YP=P[1], Food=Food_quantity, Food_width=Food_width) # gets first obs and advantages
            Food_quantity -= Eaten*eating_speed
            
            Agent.initialize(obs)
            done = False
            # -----------------------------------------
            for step in range(n_max_steps):
                count += 1
                actions, logp = Agent.get_actions(flag_logp=True) #return actions vector to give particles, and label
                
                if ((starting_food > 0) and (Food_quantity < 20)):
                    Food_width = 0
                    wait -= 1
                    if (wait == 0):
                        theta += np.random.rand()*np.pi*2
                        displ = np.random.normal(loc=150, scale=25)
                        P += displ*np.array([np.cos(theta),np.sin(theta),0.])
                        Food_quantity = starting_food
                        Food_width = starting_food_width
                        wait = 100
                        
                obs, rewards, Eaten, done, info = md.evolve_MD(actions.astype(int), XP=P[0], YP=P[1], Food=Food_quantity, Food_width=Food_width) #evolve systems from given actions
                Food_quantity -= Eaten*eating_speed
     
                md.print_xyz_food_actions(P[0], P[1], Food_quantity, Food_width, logp, actions.astype(int))

                if (step == n_max_steps-1):
                    done = True

                if ((step>0) and (count%steps_update == 0)):
                    Agent.add_env_timeframe([], obs, rewards, done)
                    Agent.train_step(epochs=50)
                    Agent.initialize(obs)
                else:
                    Agent.add_env_timeframe([], obs, rewards, done)

                order,  swirl = md.get_order()
                print('{} {} {} {} {} {} {} {}'.format(iMD, step, np.sum(rewards), P[0], P[1], Food_quantity, order, swirl))

            if (iMD%20 == 19):
                # INTERMEDIATE SAVES.
                Agent.save_models(path=models_rootname, final_save = True)

    # LAST SAVE.
    Agent.save_models(path=models_rootname, final_save = True)

