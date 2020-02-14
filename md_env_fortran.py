# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
import time
import evolve
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
    def __init__(self, md_type, index=0, N=10, size=10, steps=20, vel=0.5, dt=0.2, torque=25.0):

        self.dt = dt
        self.vel_prey = vel
        self.torque_prey = 1.0 / 350.0 * torque # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)
        
        self.N = N
        self.size = size
        self.n_MD_steps =steps

        self.Dt = 0.014
        self.Dr = 1.0 / 350.0
        self.Rm = math.sqrt(2*self.Dt/self.dt)
        self.Rr = math.sqrt(2*self.Dr/self.dt)

        self.filexyz='traj'+str(index)+'.xyz'
        self.particles = self.reinitialize_random_for_MD(index)
        self.md_type = md_type
        assert self.md_type in ['group', 'demix'], 'MD type not recognized'
        
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
    def print_xyz(self):
        p = self.particles
        xyz_file = open(self.filexyz, "a") 
        xyz_file.write('\n\n')
        for i in range(self.N):
            if self.md_type == 'demix':
                xyz_file.write(str(i//(self.N/2))+' '+str(p[i,0])+' '+str(p[i,1])+' 0.0 '+str(np.cos(p[i,2]))+' '+str(np.sin(p[i,2]))+'\n')
            #xyz_file.write('p '+str(predator[0,0])+ ' '+str(predator[0,1])+' 0.0 ' + str(predator[0,2]) + '\n')
            elif self.md_type == 'group':
                xyz_file.write('0 '+str(p[i,0])+' '+str(p[i,1])+' 0.0 '+str(np.cos(p[i,2]))+' '+str(np.sin(p[i,2]))+'\n')

    def get_o_r_demix_fortran(self):
        t0 = time.time()    
        p = self.particles 
        obs, rewards = evolve.get_o_r_demix(p[:,0],p[:,1],p[:,2],1.0,self.N) #1.0 is cost associated to having "others" in sight
        return obs, rewards

    def get_o_r_group_fortran(self):
        t0 = time.time()
        p = self.particles
        obs, rewards = evolve.get_o_r_group(p[:,0],p[:,1],p[:,2],self.N)
        return obs, rewards


    def get_obs_rewards(self):
        if self.md_type == 'group':
        	   return self.get_o_r_group_fortran()
        elif self.md_type == 'demix':
        	   return self.get_o_r_demix_fortran()
    

    def evolve_MD(self, action):
        t0 = time.time()
        done = False
        X = self.particles[:,0]        
        Y = self.particles[:,1]
        T = self.particles[:,2]
        self.particles = evolve.evolve_md(X, Y, T, action, self.Rm, self.Rr, self.dt, self.n_MD_steps, self.torque_prey, self.vel_prey, self.N)
        obs, rewards = self.get_obs_rewards()
        return obs, rewards, done, {}
#
# ------------------------------------
# End of class MD
# ------------------------------------

# CHECK FOR CONE OF SIGHT
def minus_0p5pi_to_1p5pi_mod(theta1):
    return (theta1 + math.pi)%(2*math.pi)-math.pi/2.0

def get_cone_sight(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    dist = math.sqrt(dx*dx + dy*dy)
    dist_theta = math.atan2(dy, dx)
    rel_theta = ((dist_theta - A[2])/math.pi + 1.0)%(2.0)-0.5
    n_cone = math.floor(rel_theta*5) 
    return int(n_cone), dist
	
