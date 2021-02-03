# TEST WITH MD
import numpy as np
import sys
from md_env_fortran_rod import MD_ROD
from firstrl import AgentActiveMatter
from scipy.stats import entropy as entropy
import scipy
import time
from tqdm import tqdm


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


# ------------
mode = 3
n_input = 10
# ------------

n_actions = 4
gam = 0.95
lam = 0.97
CL =0.03
en_coeff=0.01



vel_act = 0.35
vel_tor = 0.20


N = 30 #number of particles
steps_update = 64

# -- FILL FOR SIMULATION -------------------------

skew = False
n_MD = 20
total_time = 3600
mu_K = 0 # FULL PARALLEL FORCES TO THE ROD!
massRod = 0.1 # SCALE OF FRICTION
step_time = 5
torque = 25
start_MD = 0


# ------------------------------------------------

restart = False
if (start_MD > 0):
    restart = True
models_rootname = 'model_sim_NoSkew_T{}_M{}_muK{}_{}sAct'.format(torque, massRod, mu_K, step_time)

Agent = AgentActiveMatter(input_dim=n_input, output_dim=n_actions, en_coeff=en_coeff, CL=CL, gamma=gam, models_rootname=models_rootname, lam=lam, lrP=0.0005, lrV=0.001,  restart_models=restart)

# ------------------------------------------------

dt = 0.1
steps = int(step_time/dt)
n_max_steps = int(total_time/step_time)


for iMD in  tqdm(range(start_MD, start_MD + n_MD)):
        # reinitialize the class MD with the new index
        traj_flag=False
        if (iMD%1 == 0):
            traj_flag=True
        md = MD_ROD(index=iMD, N=N, size=100, skew=skew, 
		steps=steps, vel_act=vel_act, vel_tor=vel_tor, dt=dt, torque=torque, 
		sizeRod=96, massRod=massRod,
        	distRod=1.6, ext_rod=1., cen_rod=1.,
		obs_type='1overR', cones=5, cone_angle=180., flag_side=False,
		flag_LOS=False, ss=0.01, ssrod=0.01, mu_K = mu_K,
		traj=traj_flag, mode=mode)
        print('\n\n\n #NEW MD INITIALIZATION!')
        obs, rewards = md.get_obs_rewards() # gets first obs and advantages
        Agent.initialize(obs)
        done = False
        # -----------------------------------------
        for step in range(n_max_steps):
            
            #if (iMD == n_MD-1):
            #    np.savetxt(obs_file, obs)
            actions, logp = Agent.get_actions(flag_logp=True) #return actions vector to give particles, and label
            obs, rewards, done, info = md.evolve_MD(actions.astype(int)) #evolve systems from given actions
            #print(obs)
            md.print_xyz_actions(actions, logp)
            #print(obs[0,2],obs[0,7])
            if ((step>0) and (step%steps_update == 0)):
                lost = [i for i in range(obs.shape[0])]
                Agent.add_env_timeframe(lost, obs, rewards, done)
                Agent.train_step(epochs=50)
                Agent.initialize(obs)
            else:
                Agent.add_env_timeframe([], obs, rewards, done)
            print('{} {} {}'.format(iMD, step, np.sum(rewards)))

Agent.save_models(path=models_rootname, final_save = True)

