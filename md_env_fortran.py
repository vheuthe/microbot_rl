# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
import time
import evolve_fortran as evolve
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
    def __init__(self, md_type, index=0, N=10, size=10, steps=20, vel_act=0.35, vel_tor=0.20, dt=0.2, torque=25.0, cost=1., traj=True):

        
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
        
        
        # output trjectory
        self.traj = traj
        if (self.traj):
            self.filexyz='traj'+str(index)+'.xyz'
        self.filexyz='traj'+str(index)+'.xyz'
        self.particles = self.reinitialize_random_for_MD(index)
        self.md_type = md_type
        assert self.md_type in ['group', 'mix', 'demix', 'switch'], 'MD type not recognized'
        
        # Observables and reward functions & parameters
        self.cost = cost
        if (md_type in ['group']):
            self.Nobs = 10
        if (md_type in ['mix']):
            self.Nobs = 10
            self.mode = 1
        if (md_type in ['demix']):
            self.Nobs = 10
            self.mode = 2
        if (md_type in ['switch']):
            self.Nobs = 12
            self.mode = 3
        
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

  # PRINT TRAJECTORY
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
                elif self.md_type in ['group']:
                    xyz_file.write('P {} {} 0.0 {} {} {}\n'.format(p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i]))

    def get_o_r_switch_task_fortran(self, switch=-1):
        t0 = time.time()    
        p = self.particles 
        assert switch >= 0
        obs, rewards = evolve.get_o_r_mix_tasks(p[:,0], p[:,1], p[:,2], self.cost, self.mode, switch, self.Nobs, self.N) #self.cost is cost associated to having "others" in sight
        return obs, rewards
        
    def get_o_r_mix_tasks_fortran(self):
        t0 = time.time()    
        p = self.particles 
        obs, rewards = evolve.get_o_r_mix_tasks(p[:,0], p[:,1], p[:,2], self.cost, self.mode, -1, self.Nobs, self.N) #1.0 is cost associated to having "others" in sight. -1 is fake switch
        return obs, rewards

    def get_o_r_group_fortran(self):
        t0 = time.time()
        p = self.particles
        obs, rewards = evolve.get_o_r_group_task(p[:,0],p[:,1],p[:,2],self.N)
        return obs, rewards

    def get_obs_rewards(self, switch=-1):
        if self.md_type == 'group':
        	return self.get_o_r_group_fortran()
        elif (self.md_type == 'switch'):
        	return self.get_o_r_switch_task_fortran(switch)
        elif (self.md_type in ['demix', 'mix']):
        	return self.get_o_r_mix_tasks_fortran()
    

    def evolve_MD(self, action, switch=-1):
        t0 = time.time()
        done = False
        X = self.particles[:,0]        
        Y = self.particles[:,1]
        T = self.particles[:,2]
        self.particles = evolve.evolve_md(X, Y, T, action, self.Rm, self.Rr, self.dt, self.n_MD_steps, self.torque, self.vel_act, self.vel_tor, self.N)
        obs, rewards = self.get_obs_rewards(switch)
        self.rewards = rewards
        return obs, rewards, done, {}
#
# ------------------------------------
# End of class MD
# ------------------------------------
