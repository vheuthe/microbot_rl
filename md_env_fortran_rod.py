# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
from scipy.stats import entropy
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
    def __init__(self, index=0, N=10, size=10, 
                steps=20, vel_act=0.35, vel_tor=0.2, dt=0.2, torque=25.0, 
                sizeRod=3, massRod=10., inertiaRod=1., distRod=2., ext_rod=1.0, cen_rod=1.0,
                obs_type=1, cones=5, cone_angle=180., flag_side=True, flag_LOS=True,
                ss=6.2, ssrod=0.0,
                traj=False, mode=1):

        # internal knowledge of system
        self.N = N
        self.size = size

        # sight characterization:
        # obs_type -> 1/r, 1/r^2
        # cones -> number of cones
        # cone angle -> [-cone_angle, cone_angle] sight, in deg, converted in rad
        # flag_side -> whether particles can see through the rod!
        assert obs_type in ['1overR', '1overR2'], 'Obs Type not Recognized'
        if obs_type=='1overR': 
            self.obs_type = 1
        elif obs_type=='1overR2':
            self.obs_type = 2
        self.cones = cones
        self.cone_angle = cone_angle / 180. * np.pi
        self.flag_side = int(flag_side)
        self.flag_LOS = int(flag_LOS)
        self.Nobs = cones*(2+flag_side)
        
        assert (not (flag_side and flag_LOS)), 'Having LOS and view across rod together makes no sense.'

        # Rod geometry parameters
        # total lenght "sizeRod" and bead distance "distRod" dictates number of beads "Nrod"
        # if sizeRod not exactly multiple of distRod, sizeRod is conserved.
        self.sizeRod = sizeRod
        self.Nrod = int(sizeRod / distRod + 1)
        self.distRod = sizeRod / (self.Nrod - 1)
        self.massRod = massRod # total mass of object
        self.inertiaRod = inertiaRod # ratio of equivalent rigid body inertia, NOT true inertia
        self.ext_rod = ext_rod
        self.cen_rod = cen_rod
        self.ssrod = ssrod #if initialized to 0, it is automatically calculated in evolve_fortran_rod subroutine
        self.ss = ss

        # type of task.
        # determines reward function, and observation space.
        # 1 - move rod
        # 2 - move rod along -x direction
        # 3 - rotate rod
        self.rewards = np.zeros(N)
        self.mode = mode
        if (self.mode == 2): #directional pushing
            self.Nobs += 2
        if (self.mode == 4): #rotation with direction s      
            self.Nobs += 1
        
        # parameters of dynamics
        self.n_MD_steps = steps
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
        self.particles, self.rod = self.reinitialize_random_for_MD(index)
        self.old_rod = self.rod

# --------------------------
# INITIALIZE RANDOMLY X,Y IN A SQUARE LATTICE AND THETA [-pi, pi] 
    def reinitialize_random_for_MD(self, index):
        sN = np.int(np.sqrt(self.N))+1
        particles = np.random.rand(self.N, 3)*[0.0,0.0,2*np.pi]
        pos = np.array([[i+0.5,j,0] for i in np.arange(-sN//2-2,sN//2+1) for j in np.arange(-sN//2-1,sN//2+1)])
        for i in range(sN):
            for j in range(sN):
                if (i*sN+j) < self.N :
                  oo = np.random.randint(pos.shape[0])
                  particles[i*sN+j,:] += pos[oo]*10.0
                  pos = np.delete(pos, oo, axis=0)
        particles[particles[:,0] <= 0]  -= [10.0, 0, 0]
        particles[particles[:,0] > 0]  += [10.0, 0, 0]
        rod = np.array([[0.0, (i-(self.Nrod-1)/2)*self.distRod] for i in np.arange(self.Nrod)])
        if (self.traj):
            open(self.filexyz, "w")
        return particles, rod

  # PRINT TRAJECTORY
    def print_xyz(self):
        if (self.traj):
            p = self.particles
            rod = self.rod
            olr = self.old_rod
            deltacm = np.mean(rod, axis=0) - np.mean(olr, axis=0)
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            for i in range(self.N):
                xyz_file.write('0 {} {} 0.0 {} {} {} {}\n'.format( p[i,0], p[i,1], np.cos(p[i,2]), np.sin(p[i,2]), self.rewards[i], 6.2 ) )
            for i in range(self.Nrod):
                sigma_rod = 6.2*(self.cen_rod + (self.ext_rod-self.cen_rod)*abs(i/(self.Nrod*1.)-0.5))
                xyz_file.write('1 {} {} 0.0 {} {} 0.0 {}\n'.format(rod[i,0], rod[i,1], deltacm[0], deltacm[1], sigma_rod))
                
                
    def print_xyz_actions(self, actions, logp):
        if (self.traj):
            p = self.particles
            rod = self.rod
            olr = self.old_rod
            deltacm = np.mean(rod, axis=0) - np.mean(olr, axis=0)
            # calculate probability of different actions
            prob = np.exp(logp)
            s_entropy = entropy(prob, axis=1)
            # 
            xyz_file = open(self.filexyz, "a") 
            xyz_file.write('\n\n')
            for i in range(self.N):
                xyz_file.write('0 {} {} 0.0 {} {} {} {} {} {} {} {} {} {}\n'.format( p[i,0], p[i,1], np.cos(p[i,2]), 
                np.sin(p[i,2]), self.rewards[i], 6.2, actions[i], prob[i,0], prob[i,1], prob[i,2], prob[i,3], s_entropy[i] ) )
            for i in range(self.Nrod):
                sigma_rod = 6.2*(self.cen_rod + (self.ext_rod-self.cen_rod)*abs(i/(self.Nrod*1.)-0.5))
                xyz_file.write('1 {} {} 0.0 {} {} 0.0 {}\n'.format(rod[i,0], rod[i,1], deltacm[0], deltacm[1], sigma_rod))

  # CALLS THE FORTRAN SUBROUTINE FOR OBS AND REWARDS IN PRESENCE OF A ROD
    def get_o_r_rod_fortran(self, rotDir=0, old_rotDir=0, flag_side=0, obs_type=1):
        t0 = time.time()    
        p = self.particles 
        r = self.rod
        olr = self.old_rod
        if (self.mode == 4):
            assert rotDir in [-1,1]
            assert old_rotDir in [-1,1]
        obs, rewards = evolve.get_o_r_rod(p[:,0],p[:,1],p[:,2], 
                                          r[:,0], r[:,1], olr[:,0],olr[:,1], 
                                          self.mode, rotDir, old_rotDir, 
                                          flag_side, self.flag_LOS, 
                                          self.ss, self.ssrod,
                                          obs_type, 
                                          self.cones, self.cone_angle, 
                                          self.Nobs, self.N, self.Nrod)
        self.rewards = rewards
        return obs, rewards

    def get_obs_rewards(self, rotDir=0, old_rotDir=0):
        return self.get_o_r_rod_fortran(rotDir, old_rotDir, self.flag_side, obs_type=self.obs_type)

    def evolve_MD(self, action, rotDir=0, old_rotDir=0):
        t0 = time.time()
        done = False
        X = self.particles[:,0]        
        Y = self.particles[:,1]
        T = self.particles[:,2]
        Xrod = self.rod[:,0]
        Yrod = self.rod[:,1]
        mRod = self.massRod
        IRod = self.inertiaRod
        self.old_rod = self.rod
        self.particles, self.rod = evolve.evolve_md_rod(mRod, IRod, 
                                    X, Y, T,
                                    Xrod, Yrod, self.distRod, action, 
                                    self.Rm, self.Rr, self.dt, self.n_MD_steps, 
                                    self.torque, self.vel_act, self.vel_tor, 
                                    self.ext_rod, self.cen_rod,self.N, self.Nrod)
        obs, rewards = self.get_obs_rewards(rotDir, old_rotDir)
        return obs, rewards, done, {}
#
# ------------------------------------
# End of class MD
# ------------------------------------
	
