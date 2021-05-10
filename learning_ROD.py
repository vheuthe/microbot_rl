# Reworked analogous to Robert's learning_food.py, where
# executing an episode, step, etc. are all functions.
# TEST WITH MD
import numpy as np
import sys
import scipy
import time
from scipy.stats import entropy as entropy

from environments.rod import MD_ROD
from firstrl import AgentActiveMatter


default_parameters = {

    # RL Agent
    'input_dim': 10,
    'output_dim': 4,
    'en_coeff': 0.01,
    'CL': 0.03,
    'gamma': 0.95,
    'lam': 0.97,
    'lrPI': 0.0005,
    'lrV': 0.001,
    'target_kl': 0.02,
    'model_structure': [(32, 'relu'),(16, 'relu'),(16, 'relu')],

    # For Rewards
    'close_pen': 0.6, # Prefactor for closeness penalty
    'rotRewFact': 2, # Prefactor for rotation rewards for rewards based on forces
    'pushRewFact': 3,
    'rewMode': 'primitive', # Mode of rewards ('forces', 'primitive' or 'classic')
    'rewCutoff': 30, # float(sys.argv[1]), # 8, # Cutoff for the primitive rewards

    # Particles
    'vel_act': 0.35,
    'vel_tor': 0.2, # Velocity during rotation (particles do not stand still)
    'N': 30,
    'torque': 25,
    'ss': 0.01,
    'obs_type': '1overR',
    'conse': 5,
    'cone_angle': 180,
    'flag_side': False,
    'flag_LOS': False,

    # Rod
    'ss_rod': 0.01,
    'mu_K': 1,
    'sizeRod': 96,
    'distRod': 1.6,
    'ext_rod': 1.,
    'cen_rod': 1.,

    # For MD part of simulation
    'steps_update': 128,
    'steps': 20,
    'n_MD': 300, # 100,
    'total_time': 3600,
    'step_time': 5,
    'dt': 0.1,
    'start_MD': 0,
    'mode': 3, # 3: normal rotation, 4: rotation in direction s, 2: directional pushing, 6:push along long direction
    'skew': False,
    'size': 100,
    'traj': True
}

parameters = default_parameters

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

    data_path = '/data/scc/veit-lorenz.heuthe/primitive_reward_test' # '/mnt/c/Users/veit/Documents/PhD/SimData/test_primitive_reward/check'

    # ------------------------------------------------

    load_models = None
    models_rootname = data_path + '/model_sim_rot_{}_{}sAct'.format(parameters["mu_K"], parameters["step_time"])
    if (parameters["start_MD"] > 0):
        load_models = models_rootname

    Agent = AgentActiveMatter(
        models_rootname = 'a',
        restart_models = False,
        **parameters)

    # ------------------------------------------------

    steps = int(parameters["step_time"]/parameters["dt"])
    n_max_steps = int(parameters["total_time"]/parameters["step_time"])


    for iMD in range(parameters["start_MD"], parameters["start_MD"] + parameters["n_MD"]): # Episodes -> each is one training run
            # reinitialize the class MD with the new index
            traj_flag=False
            if (iMD%1 == 0):
                traj_flag=True
            md = MD_ROD(
                index=iMD, **parameters,
                data_path=data_path)
            print('\n\n\n #NEW MD INITIALIZATION!')
            obs, rewards = md.get_obs_rewards() # gets first obs and advantages
            Agent.initialize(obs)
            # -----------------------------------------
            for step in range(n_max_steps):

                #if (iMD == n_MD-1):
                #    np.savetxt(obs_file, obs)
                actions, logp = Agent.get_actions() #return actions vector to give particles, and label
                obs, rewards, done, info = md.evolve_MD(actions) #evolve systems from given actions
                #print(obs)
                md.print_xyz_actions(actions, logp)
                #print(obs[0,2],obs[0,7])
                if ((step>0) and (step%parameters['steps_update'] == 0)):
                    lost = [i for i in range(obs.shape[0])]
                    Agent.add_environment_response(lost, obs, rewards)
                    Agent.train_step(epochs=50)
                    Agent.initialize(obs)
                else:
                    Agent.add_environment_response([], obs, rewards)
                print('{} {} {}'.format(iMD, step, np.sum(rewards)))

    Agent.save_models(path=models_rootname)
