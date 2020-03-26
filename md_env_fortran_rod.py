# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
import time
import evolve_fortran_rod as evolve
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

class MD_ROD():
    def __init__(self, index=0, N=10, Nrod=3, size=10, steps=20, vel=0.5, dt=0.2, torque=25.0, massRod=10., traj=False, mode=1, rotDirection=0):

        # internal knowledge of system
        self.N = N
        self.Nrod = Nrod
        self.size = size
        self.n_MD_steps = steps
        # cone of sight. 5x particles, 5x rod particles
        self.Nobs = 10
        self.rewards = np.zeros(N)

        # type of task.
        # determines reward function, and observation space.
        # 1 - move rod
        # 2 - move rod along -x direction
        # 3 - rotate rod
        self.mode = mode
        if (self.mode == 2): #directional pushing
            self.Nobs = 12
        
        if (self.mode == 4): #rotation with direction    
            self.
            self.Nobs = 11
        
        # parameters of dynamics
        self.dt = dt
        self.Dt = 0.014
        self.Dr = 1.0 / 350.0
        self.Rm = math.sqrt(2*self.Dt/self.dt)
        self.Rr = math.sqrt(2*self.Dr/self.dt)
        self.massRod = massRod
        self.vel_prey = vel
        self.torque_prey = 1.0 / 350.0 * torque # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)
        
        # output trjectory
        self.traj = traj
        if (self.traj):
            self.filexyz='traj'+str(index)+'.xyz'
        self.particles, self.rod = self.reinitialize_random_for_MD(index)
        self.old_rod = self.rod

# --------------------------
# INITIALIZE RANDOMLY X,Y IN A SQUARE LATTICE AND THETA [-pi, pi] 
    def reinitialize_random_for_MD(self, index):
        sN = np.int(np.sqrt(self.N))+1
        particles = np.random.rand(self.N, 3)*[0.0,0.0,2*np.pi]
        pos = np.array([[i,j,0] for i in np.arange(-sN//2-1,sN//2+1) for j in np.arange(-sN//2-1,sN//2+1)])
        for i in range(sN):
            for j in range(sN):
                if (i*sN+j) < self.N :
                  oo = np.random.randint(pos.shape[0])
                  particles[i*sN+j,:] += pos[oo]*5.0
                  pos = np.delete(pos, oo, axis=0)
        rod = np.array([[(sN+1.0)*5.0, (i-(self.Nrod-1)/2.0)*1.] for i in np.arange(self.Nrod)])
        if (self.traj):
            open(self.filexyz, "w")
        return particles, rod

  # PRINT TRAJECTORY
    def print_xyz(self):
        if (self.traj):
            p = self.particles
            rod = self.rod
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            for i in range(self.N):
                xyz_file.write('0 {} {} 0.0 {} {} {}\n'.format( p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i] ) )
            for i in range(self.Nrod):
                xyz_file.write('1 {} {} 0.0 0.0 0.0 0.0\n'.format(rod[i,0], rod[i,1]) )

  # CALLS THE FORTRAN SUBROUTINE FOR OBS AND REWARDS IN PRESENCE OF A ROD
    def get_o_r_rod_fortran(self):
        t0 = time.time()    
        p = self.particles 
        r = self.rod
        olr = self.old_rod
        obs, rewards = evolve.get_o_r_rod(p[:,0],p[:,1],p[:,2], r[:,0], r[:,1], olr[:,0],olr[:,1], self.mode, self.Nobs, self.N, self.Nrod)
        # DEGUB
        self.rewards = rewards
        return obs, rewards

    def get_obs_rewards(self):
        return self.get_o_r_rod_fortran()

    def evolve_MD(self, action):
        t0 = time.time()
        done = False
        X = self.particles[:,0]        
        Y = self.particles[:,1]
        T = self.particles[:,2]
        Xrod = self.rod[:,0]
        Yrod = self.rod[:,1]
        mRod = self.massRod
        self.particles, self.rod = evolve.evolve_md_rod(mRod, X, Y, T, Xrod, Yrod, action, self.Rm, self.Rr, self.dt, self.n_MD_steps, self.torque_prey, self.vel_prey, self.N, self.Nrod)
        obs, rewards = self.get_obs_rewards()
        self.old_rod = self.rod
        return obs, rewards, done, {}
#
# ------------------------------------
# End of class MD
# ------------------------------------
	
