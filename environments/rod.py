# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from math import e
import time
from fortran import evolve_rod_rigid as evolve
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
    def __init__(self, index=0, N=10, size=10, skew=False,
                nStepSim=20, vel_act=0.35, vel_tor=0.2, dt=0.2, torque=25.0,
                sizeRod=3, massRod=10., inertiaRod=1., distRod=2., Nrod=60, ext_rod=1.0, cen_rod=1.0, mu_K=0.0,
                Dt=0.014, Dr=1.0 / 350.0,
                obs_type=1, cones=5, cone_angle=180., flag_side=True, flag_LOS=True,
                ss=6.2, ssrod=0.0, ss_touch=6.8, mode=1, swirl=False,
                data_path=None, rewMode='classic',
                close_pen=0, rotRewFact=2, pushRewFact=3,
                rewCutoff=8, startConfig='standard', **unused_parameters):

        # path for writing the trajectories
        self.data_path = data_path
        # internal knowledge of system
        self.skew = skew
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
        #print('AT _INIT_ Nobs = {}'.format(self.Nobs))

        assert (not (flag_side and flag_LOS)), 'Having LOS and view across rod together makes no sense.'

        # Rod geometry parameters
        # total lenght "sizeRod" and bead distance "distRod" dictates number of beads "Nrod"
        # if sizeRod not exactly multiple of distRod, sizeRod is conserved.
        self.sizeRod = sizeRod
        self.Nrod = Nrod
        self.distRod = sizeRod / (Nrod -1)
        self.massRod = massRod # total mass of object
        self.inertiaRod = inertiaRod # ratio of equivalent rigid body inertia, NOT true inertia
        self.ext_rod = ext_rod
        self.cen_rod = cen_rod
        self.ssrod = ssrod #if initialized to 0, it is automatically calculated in evolve_fortran_rod subroutine
        self.ss = ss
        self.mu_K = mu_K # kinetic friction - like along rod.
        self.close_pen = close_pen # factor for penalizing closenes (nearest neighbor)

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
        if (self.mode == 6): #push along long direction
            self.Nobs += 1

        self.rotRewFact = rotRewFact # These are factors for the implementation of rewards based on forces
        self.pushRewFact = pushRewFact
        self.rewMode = rewMode # 'forces' or 'classic'
        self.rewCutoff = rewCutoff # for primitive rewards: the max distance to the rod that still gets rewarded
        self.startConfig = startConfig # which configuration to start with

        # parameters of dynamics
        self.nStepSim = nStepSim # number of integration steps done in every simulation step
        self.dt = dt
        self.Dt = Dt # 0.014
        self.Dr = Dr # 1.0 / 350.0
        self.Rm = math.sqrt(2*self.Dt/self.dt)
        self.Rr = math.sqrt(2*self.Dr/self.dt)
        self.vel_act = vel_act
        self.vel_tor = vel_tor
        self.torque = 1.0 / 350.0 * torque # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)

        if self.startConfig == 'standard':
            self.particles, self.rod = self.reinitialize_random_for_MD(swirl)
        elif self.startConfig == 'test_friction':
            self.particles, self.rod = self.reinitialize_test_friction(swirl)
        elif self.startConfig == 'biased':
            self.particles, self.rod = self.reinitialize_biased(swirl)


        self.old_rod = np.zeros(self.rod.shape)
        self.old_rod[:] = self.rod[:]
        self.Particle_perf = np.zeros((self.particles.shape[0], 1))
        self.part_rod_forces = np.zeros((self.particles.shape[0], 3))

# --------------------------
# INITIALIZE RANDOMLY X,Y IN A SQUARE LATTICE AND THETA [-pi, pi]
    def reinitialize_random_for_MD(self, swirl=False):

        sN = np.int(np.sqrt(self.N))+1

        if not swirl:
            particles = np.random.rand(self.N, 3)*[0.0,0.0,2*np.pi]
            if (self.skew):
                pos = np.array([[i+0.5,j,0] for i in np.arange(1,sN+2) for j in np.arange(-sN//2-1,sN//2+1)]) # ONLY ON RIGHT SIDE
            else:
                pos = np.array([[i+0.5,j,0] for i in np.arange(-sN//2-2,sN//2+1) for j in np.arange(-sN//2-1,sN//2+1)])

            for i in range(sN):
                for j in range(sN):
                    if (i*sN+j) < self.N :
                      oo = np.random.randint(pos.shape[0])
                      particles[i*sN+j,:] += pos[oo]*10.0
                      pos = np.delete(pos, oo, axis=0)
        else:
            Lrod = self.Nrod*self.distRod
            particles = np.zeros((self.N,3))
            particles[:self.N//2,2] = np.pi/2
            particles[self.N//2:,2] = -np.pi/2
            particles[:self.N//2,0] = 1
            particles[self.N//2:,0] = -1
            particles[:self.N//2,1] = np.arange(-Lrod/2, Lrod/2, Lrod*2/(self.N-1))
            particles[self.N//2:,1] = np.arange(-Lrod/2, Lrod/2, Lrod*2/(self.N-1))

        particles[particles[:,0] <= 0] -= [10.0, 0, 0]
        particles[particles[:,0] > 0] += [10.0, 0, 0]


        rod = np.array([[0.0, (i-(self.Nrod-1)/2)*self.distRod] for i in np.arange(self.Nrod)])

        return particles, rod

  # Initialize with two particles close to the rod and at one end of it (for test_friction)

    def reinitialize_test_friction(self, swirl=False):
        # Initializes with two particles close to the rod at one of its ends
        particles = np.array([[-7, -40, np.pi/2-np.pi/4], [7, -40, np.pi/2+np.pi/4]])

        rod = np.array([[0.0, (i-(self.Nrod-1)/2)*self.distRod] for i in np.arange(self.Nrod)])

        return particles, rod


    def reinitialize_biased(self, swirl=False):
        # Initializes with the partielces on opposite sides at the ends of the rod and pointing towards it
        gridSz = np.int(np.sqrt(self.N/2)) + 1
        gridVals = [10 * (element + 0.5 - gridSz/2) for element in list(range(gridSz))]

        grid0 = np.array(gridVals, ndmin=2) + 1j * np.transpose(np.array(gridVals, ndmin=2)) # Prototype of the grid
        grid1 = grid0 + (10 * (1  + gridSz/2) + 1j * (self.sizeRod/2 - 10 * gridSz/2)) # Grid shifted to the upper right end of the rod
        grid2 = grid0 + (-10 * (1  + gridSz/2) - 1j * (self.sizeRod/2 - 10 * gridSz/2)) # Grid shifted to the lower left end of the rod

        particles = np.zeros((self.N,3))

        for i in range(self.N):
            if i<grid1.size:
                particles[i,:] = [np.real(grid1[np.unravel_index(i, grid1.shape)]), \
                                  np.imag(grid1[np.unravel_index(i, grid1.shape)]), \
                                  -np.pi]
            else:
                particles[i,:] = [np.real(grid2[np.unravel_index(i - grid1.size, grid2.shape)]), \
                                  np.imag(grid2[np.unravel_index(i - grid1.size, grid2.shape)]), \
                                  0]

        rod = np.array([[0.0, (i-(self.Nrod-1)/2)*self.distRod] for i in np.arange(self.Nrod)])

        return particles, rod



    def evolve_MD(self, action, rotDir=0, old_rotDir=0):
        '''
        Evolves the particle and rod situation one simulation step (=nStepSim times one dt time-step)
        '''

        X = self.particles[:,0]
        Y = self.particles[:,1]
        T = self.particles[:,2]
        Xrod = self.rod[:,0]
        Yrod = self.rod[:,1]
        mRod = self.massRod
        IRod = self.inertiaRod
        self.old_rod[:] = self.rod[:]
        self.particles, self.rod, self.part_rod_forces = evolve.evolve_md_rod(mRod, IRod,
                                    X, Y, T,
                                    Xrod, Yrod, self.distRod, action,
                                    self.Rm, self.Rr, self.dt, self.nStepSim,
                                    self.torque, self.vel_act, self.vel_tor,
                                    self.ext_rod, self.cen_rod, self.mu_K,
                                    self.N, self.Nrod)
        obs, rewards = self.get_obs_rewards(rotDir, old_rotDir)

        # Calculating the rod orientation for saving it to the stats file
        rodTheta = np.angle(complex(self.rod[-1,0] - self.rod[0,0], self.rod[-1,1] - self.rod[0,1]))

        return obs, rewards, rodTheta



  # CALLS THE FORTRAN SUBROUTINE FOR OBS AND REWARDS IN PRESENCE OF A ROD
    def get_obs_rewards(self, rotDir=0, old_rotDir=0, flag_side=0, obs_type=1):
        '''
        This calculates a reward based on the contribution of each particle to the performance.
        The contribution of each particle is estimated by evaluating how well the forces the
        particle exerted on the rod meet the desired rod.
        '''
        # Determining the performance P of each particle (this is the important part)
        r = self.rod
        olr = self.old_rod
        p = self.particles
        if (self.mode == 4):
            assert rotDir in [-1,1]
            assert old_rotDir in [-1,1]

        # Now the observables are determined, if rewMode=='classic' the rewards determined here are used, too
        obs, rewards, self.touch = evolve.get_o_r_rod(p[:,0], p[:,1], p[:,2],
                                          r[:,0], r[:,1], olr[:,0], olr[:,1],
                                          self.mode, rotDir, old_rotDir,
                                          flag_side, self.flag_LOS,
                                          self.ss, self.ssrod, self.massRod,
                                          self.ext_rod, self.cen_rod,
                                          obs_type,
                                          self.cones, self.cone_angle, self.close_pen,
                                          self.Nobs, self.N, self.Nrod)

        if self.rewMode == 'classic':

            self.rewards = rewards

        elif self.rewMode == 'forces':

            rewards = self.get_forces_rewards(absFlag=False) # Determines the rewards according to the forces and te current mode

            self.rewards = rewards

        elif self.rewMode == 'absForces':

            rewards = self.get_forces_rewards(absFlag=True) # Determines the rewards according to the forces and te current mode

            self.rewards = rewards

        elif self.rewMode == 'primitive': # the primRewMode spwcifies if touching or closeness are decisive

            rewards = self.get_primitive_rewards(primRewMode='close') # Determines rewards in primitive way (close? rotated?) So far only for rot.

            self.rewards = rewards

        elif self.rewMode == 'primitiveTouch':

            rewards = self.get_primitive_rewards(primRewMode='touch') # Determines rewards in primitive way (close? rotated?) So far only for rot.

            self.rewards = rewards

        # Check if there are any NaNs in the rewards
        assert not np.isnan(rewards).any(), 'NaNs in rewards'

        return obs, rewards


    def get_forces_rewards(self, absFlag=False):
        '''
        This rewards on the basis of how well the forces the particles exerted on the rod
        comply to the objective (move the rod or rotate it, etc.)
        '''
        r = self.rod
        olr = self.old_rod

        if self.mode == 3: # Rotation
            dTheta_uncorr = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1])) - \
                np.angle(complex(olr[-1,0] - olr[0,0], olr[-1,1] - olr[0,1])) # Still can have jumps

            dTheta = dTheta_uncorr - np.floor(dTheta_uncorr/(2 * np.pi) + 0.5) * 2 * np.pi # Now the jumps are corrected

            self.Particle_perf = self.part_rod_forces[:,2] * np.sign(dTheta) # Performance is proportional torque fr rotation

            rewards = self.rotRewFact * dTheta * self.Particle_perf # Performance is only rewarded, if the rod has rotated

        elif self.mode == 6: # Longitudinal pushing
            dCM = complex(sum(r[:,0]) / self.Nrod - sum(olr[:,0]) / self.Nrod, \
                sum(r[:,1]) / self.Nrod - sum(olr[:,1]) / self.Nrod) # Center of mass motion in complex numbers

            rodTheta = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1]))

            dCM_lon = (dCM * e ** (-1j*rodTheta)).real # Longitudinal CoM motion of the rod

            # When rewMode is 'forces' (not 'absForces'), the particle performance are
            # the forces exerted on the rod in the right direction
            if not absFlag:

                part_rod_forces_complex = self.part_rod_forces[:,0] + 1j * self.part_rod_forces[:,1]

                self.Particle_perf = (part_rod_forces_complex * e ** (-1j*rodTheta)).real # Particle forces in the longitud. direction of the rod

            # When rewMode is 'absForces' (meaning absFlag=True), the particle
            # performance is just the sum of the absolute forces a particle exerts on
            # the rod, meaning it is rewarded more, if it is interacting more.
            elif absFlag:

                self.Particle_perf = np.sum(abs(self.part_rod_forces), axis=1)

            rewards = self.pushRewFact * dCM_lon * self.Particle_perf # Performance is only rewarded, if the rod has moved

            if np.isnan(rewards).any():
                z=1

        return rewards


    def get_primitive_rewards(self, primRewMode='touch'): # the primRewMode spwcifies if touching or closeness are decisive
        '''
        This simply rewards every particle that is present within a certain
        area around the rod if the rod has moved or rotated, etc.
        '''
        r = self.rod
        olr = self.old_rod
        p = self.particles

        # Determining the distances to the rod (for every mode)
        pCompl = np.array(p[:,0] + 1j * p[:,1], ndmin=2)
        rCompl = np.array(r[:,0] + 1j * r[:,1], ndmin=2)
        rodTheta = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1]))

        dist = np.transpose(abs(pCompl - np.transpose(rCompl))) # Particles are in rows with their distances to the rod in columns
        minDist = np.transpose(np.amin(dist, axis=1))
        closeEnough = minDist <= self.rewCutoff # Only the particles within the cutoff distance to the rod get rewarded

        if self.mode == 3: # Rotation
            dTheta_uncorr = rodTheta - np.angle(complex(olr[-1,0] - olr[0,0], olr[-1,1] - olr[0,1])) # Still can have jumps

            dTheta = dTheta_uncorr - np.floor(dTheta_uncorr/(2 * np.pi) + 0.5) * 2 * np.pi # Now the jumps are corrected

            if primRewMode == 'close':
                rewards = closeEnough * abs(dTheta) * self.rotRewFact # The direction of rotation does not matter.
            elif primRewMode == 'touch':
                rewards = self.touch * abs(dTheta) * self.rotRewFact # The direction of rotation does not matter.


        if self.mode == 6: # Longitudinal transport
            dCM = complex(sum(r[:,0]) / self.Nrod - sum(olr[:,0]) / self.Nrod, \
                sum(r[:,1]) / self.Nrod - sum(olr[:,1]) / self.Nrod) # Center of mass motion in complex numbers

            rodTheta = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1]))

            dCM_lon = (dCM * e ** (-1j*rodTheta)).real # Longitudinal CoM motion of the rod

            if primRewMode == 'close':
                rewards = closeEnough * abs(dCM_lon) * self.pushRewFact # The direction of rotation does not matter.
            elif primRewMode == 'touch':
                rewards = self.touch * abs(dCM_lon) * self.pushRewFact # The direction of rotation does not matter.

        return rewards
#
# ------------------------------------
# End of class MD
# ------------------------------------
