# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import time
import evolve_fortran_smooth as evolve
# ---------------------------------------

# ---------------------------------------
# EQUIVALENT OF GYM FOR MD ENVIRONMENT
# ---------------------------------------

#===============================================================================
# class MDEnv(gym.Env):
# 
#     def __init__(self):
# 
#   def step(self, action):
#     ...
#   def reset(self):
#     ...
#   def render(self, mode='human'):
#     ...
#   def close(self):
#     ...
#===============================================================================

class MD():
    def __init__(self, md_type, index=0, N=10, size=10, steps=20, vel_act=0.35, vel_tor=0.20, dt=0.2, torque=25.0, cost=1., food_rew=1.0, obs_type='1overR', cone_angle=180., dead_vision=0., flag_LOS = False, cones=5, traj=True):

        
        self.N = N
        self.rewards = np.zeros(N)

        self.size = size
        self.n_MD_steps =steps

        # Parameters of Dynamics
        self.dt = dt
        self.Dt = 0.014
        self.Dr = 1.0 / 350.0
        self.Rm = math.sqrt(2*self.Dt/self.dt)
        self.Rr = math.sqrt(2*self.Dr/self.dt)
        self.vel_act = vel_act
        self.vel_tor = vel_tor
        self.torque = 1.0 / 350.0 * torque # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)  
        
        if (obs_type == '1overR'):
            self.obs_type=1
        elif (obs_type == '1overR2'):
            self.obs_type=2
        else:
            print('error in receiving obs_type: only "1overR" and "1overR2" accepted. Got {}'.format(obs_type))
        
        
        # output trjectory
        self.traj = traj
        if (self.traj):
            self.filexyz='traj'+str(index)+'.xyz'
        self.filexyz='traj'+str(index)+'.xyz'
        self.particles = self.reinitialize_random_for_MD(index)
        self.md_type = md_type
        assert self.md_type in ['group', 'mix', 'demix', 'switch', 'food'], 'MD type not recognized'
        
        # Observables and reward functions & parameters
        self.flag_LOS = flag_LOS
        self.cone_angle = np.abs(cone_angle)/180.*np.pi # sight is [-cone_angle/2, cone_angle/2], in radiants
        self.dead_vision = np.abs(dead_vision)/180.*np.pi

        self.cost = cost
        if (md_type in ['group']):
            self.Nobs = 2*cones
            self.mode = 0
        if (md_type in ['mix']):
            self.Nobs = 2*cones
            self.mode = 1
        if (md_type in ['demix']):
            self.Nobs = 2*cones
            self.mode = 2
        if (md_type in ['switch']):
            self.Nobs = 2*cones+2
            self.mode = 3
        if (md_type in ['food']):
            self.Nobs = 3*cones
            self.mode = 4
            self.food_rew = food_rew
        
# --------------------------
# INITIALIZE RANDOMLY X,Y IN A BOX [-10:10,-10:10] AND THETA [-pi, pi] 
    def reinitialize_random_for_MD(self, index):
        sN = np.int(np.sqrt(self.N))+1
        particles = np.random.rand(self.N, 3)*[0.0,0.0,2*np.pi]
        pos = np.array([[i,j,0] for i in np.arange(-sN//2-1,sN//2+1) for j in np.arange(-sN//2-1,sN//2+1)])
        for i in range(sN):
            for j in range(sN):
                if (i*sN+j) < self.N :
                  oo = np.random.randint(pos.shape[0])
                  particles[i*sN+j,:] += pos[oo]*10.0
                  pos = np.delete(pos, oo, axis=0)
        open(self.filexyz, "w") 
        return particles

#------------------------------------------------------
#--- PRINT TRAJECTORY
    def print_xyz(self, switch=-1):
        if (self.traj):
            p = self.particles
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            for i in range(self.N):
                if (self.md_type in ['demix', 'mix']):
                    xyz_file.write('{} {} {} 0.0 {} {} {}\n'.format(i//(self.N/2), p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i]))
                elif (self.md_type in ['switch']):
                    xyz_file.write('{} {} {} 0.0 {} {} {} {}\n'.format(i//(self.N/2), p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), switch, self.rewards[i]))
                elif self.md_type in ['group', 'food']:
                    xyz_file.write('P {} {} 0.0 {} {} {}\n'.format(p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i]))

#--- PRINT TRAJECTORY
    def print_xyz_food(self, Xfood, Yfood, Food, Food_width):
        if (self.traj):
            p = self.particles
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            xyz_file.write('1 {} {} 0.0 {} {} {} {}\n'.format(Xfood, Yfood, 0, 0, Food, Food_width/2))
            for i in range(self.N):
                xyz_file.write('0 {} {} 0.0 {} {} {} {}\n'.format(p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i], 6.2))


#--- PRINT TRAJECTORY
    def print_xyz_food_actions(self, Xfood, Yfood, Food, Food_width, logp, actions):
        if (self.traj):
            p = self.particles
            # calculate probability of different actions
            prob = np.exp(logp)
            s_entropy = entropy(prob, axis=1)
            # 
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            xyz_file.write('1 {} {} 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0\n'.format(Xfood, Yfood, 0, 0, Food, Food_width/2))
            for i in range(self.N):
                xyz_file.write('0 {} {} 0.0 {} {} {} {} {} {} {} {} {} {}\n'.format(p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i], 6.2, actions[i], prob[i, actions[i]-1], prob[i,0], prob[i,1], prob[i,2], s_entropy[i] ))

#-------
    def print_xyz_actions(self, actions, logp, switch=-1):
        if (self.traj):
            p = self.particles
            # calculate probability of different actions
            prob = np.exp(logp)
            s_entropy = entropy(prob, axis=1)
            # 
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            for i in range(self.N):
                if (self.md_type in ['demix', 'mix']):
                    xyz_file.write('{} {} {} 0.0 {} {} {} {} {} {} {} {} {}\n'.format(i//(self.N/2), p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i], actions[i], prob[i,0], prob[i,1], prob[i,2], prob[i,3], s_entropy[i] ))
                elif (self.md_type in ['switch']):
                    xyz_file.write('{} {} {} 0.0 {} {} {} {} {} {} {} {} {} {}\n'.format(i//(self.N/2), p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), switch, self.rewards[i], actions[i], prob[i,0], prob[i,1], prob[i,2], prob[i,3], s_entropy[i]))
                elif self.md_type in ['group']:
                    xyz_file.write('P {} {} 0.0 {} {} {} {} {} {} {} {} {}\n'.format(p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i], actions[i], prob[i,0], prob[i,1], prob[i,2], prob[i,3], s_entropy[i]))
#------------------------------------------------------

#------------------------------------------------------
    def get_obs_rewards(self, switch=-1, old_switch=-1):
        if self.md_type == 'group':
        	return get_o_r_group_fortran()
        elif (self.md_type == 'switch'):
        	return self.get_o_r_switch_task_fortran(switch, old_switch)
        elif (self.md_type in ['demix', 'mix']):
        	return self.get_o_r_mix_tasks_fortran(self.obs_type)
#---------------------
    def get_o_r_group_fortran(self):
        p = self.particles
        obs, rewards = evolve.get_o_r_group_task(p[:,0],p[:,1],p[:,2],self.N)
        return obs, rewards
#---------------------
    def get_o_r_switch_task_fortran(self, switch=-1, old_switch=-1):
        p = self.particles 
        assert switch >= 0
        assert old_switch >= 0
        obs, rewards = evolve.get_o_r_mix_tasks(p[:,0], p[:,1], p[:,2], self.cost, self.mode, switch, old_switch, self.obs_type, self.cone_angle, 0, self.flag_LOS, self.Nobs, self.N) #self.cost is cost associated to having "others" in sight
        return obs, rewards
#---------------------
    def get_o_r_mix_tasks_fortran(self, obs_type):
        p = self.particles 
        obs, rewards = evolve.get_o_r_mix_tasks(p[:,0], p[:,1], p[:,2], self.cost, self.mode, -1, -1, obs_type, self.cone_angle, self.dead_vision, self.flag_LOS, self.Nobs, self.N) #1.0 is cost associated to having "others" in sight. -1 is fake switch
        return obs, rewards
#-----------------------------------------------------



#-----------------------------------------------------
    def get_obs_rewards_food(self, XP=0, YP=0, Food=0, Food_width=-1):
        if (Food_width == -1):
            Food_width = np.sqrt(Food)
        return self.get_o_r_group_food_task_fortran(XP, YP, Food, Food_width)  
#---------------------
    def get_o_r_group_food_task_fortran(self, XP, YP, Food, Food_width = -1):
        p = self.particles 
        self.dead_vision = 0
        obs, rewards, eaten = evolve.get_o_r_food_task(p[:,0], p[:,1], p[:,2], self.obs_type, self.cone_angle, self.dead_vision, self.food_rew, XP, YP, Food, Food_width, self.Nobs, self.N) 

        return obs, rewards, eaten          
#------------------------------------------------------          
        
    def get_NN(self):
        p = self.particles
        return evolve.get_neigh(p[:,0], p[:,1], self.N)

#------------------------------------------------------
    def get_order(self):
        p = self.particles 
        order, swirl = evolve.get_order_param(p[:,0], p[:,1], p[:,2], self.N)
        return order, swirl
        
#------------------------------------------------------          
    def evolve_MD(self, action, switch=-1, old_switch=-1, XP=-100., YP=-100., Food=0, Food_width=-1, flag_mobility = False):
        done = False
        X = self.particles[:,0]
        Y = self.particles[:,1]
        T = self.particles[:,2]
        self.particles = evolve.evolve_md(X, Y, T, action, self.Rm, self.Rr, self.dt, self.n_MD_steps, self.torque, self.vel_act, self.vel_tor, self.N)
        if (self.md_type in ['group', 'switch', 'demix', 'mix']):
            obs, rewards = self.get_obs_rewards(switch, old_switch)
            self.rewards = rewards
            return obs, rewards, done, {}
        elif (self.md_type in ['food']):
            obs, rewards, eaten = self.get_obs_rewards_food(XP, YP, Food, Food_width)
            self.rewards = rewards
            return obs, rewards, eaten, done, {}

#------------------------------------------------------          
      
#
# ------------------------------------
# End of class MD
# ------------------------------------
