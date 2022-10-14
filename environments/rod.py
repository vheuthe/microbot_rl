# ---------------------------------------
import numpy as np
import math
import sys
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from math import e
import time
import random
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
    def __init__(self, index=0, N=30, size=10, skew=False,
                int_steps=20, vel_act=0.35, vel_tor=0.2, dt=0.2, torque=25.0,
                fr_rod=10., inert_rod=1., l_rod=100, n_rod=60, ext_rod=1.0, cen_rod=1.0, mu_K=0.0,
                Dt=0.014, Dr=1.0 / 350.0,
                obs_type='1overR', cones=5, cone_angle=180., flag_side=False, flag_LOS=False,
                part_size=6.2, part_size_rod=0.0, part_size_touch=6.8, mode=1, swirl=False,
                data_path=None, rew_mode='WLU', prim_rew_mode='close', WLU_mode = 'non_ex', sparse_rew = False,
                close_pen=0, prox_rew=0, r_rew_fact=2, p_rew_fact=3, WLU_prefact=10000, WLU_noise='mixed',
                rew_cutoff=60, start_conf='standard', trans_dist=100,
                flag_fix_or = 0, train_ep = 100, n_rew_frames=1, **unused_parameters):

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
        self.n_obs = cones*(2+flag_side)
        #print('AT _INIT_ n_obs = {}'.format(self.n_obs))

        assert (not (flag_side and flag_LOS)), 'Having LOS and view across rod together makes no sense.'

        # Rod geometry parameters
        # total lenght "l_rod" and bead distance "dist_rod" dictates number of beads "n_rod"
        # if l_rod not exactly multiple of dist_rod, l_rod is conserved.
        self.l_rod = l_rod
        self.n_rod = n_rod
        self.dist_rod = l_rod / (n_rod -1)
        self.fr_rod = fr_rod                            # total mass of object
        self.inert_rod = inert_rod                      # ratio of equivalent rigid body inertia, NOT true inertia
        self.ext_rod = ext_rod
        self.cen_rod = cen_rod
        self.part_size_rod =part_size_rod               #if initialized to 0, it is automatically calculated in evolve_fortran_rod subroutine
        self.part_size=part_size
        self.mu_K = mu_K                                # kinetic friction - like along rod.
        self.close_pen = close_pen                      # factor for penalizing closenes (nearest neighbor)
        self.prox_rew = prox_rew                        # factor for reward for being close to the rod
        self.flag_fix_or = flag_fix_or                  # Determines, if the direction to move the rod in mode 6 is fixed to the original rod orientation or not.

        self.r_rew_fact = r_rew_fact                    # These are factors for the implementation of rewards based on forces
        self.p_rew_fact = p_rew_fact
        self.rew_mode = rew_mode                        # 'forces' or 'classic'
        self.prim_rew_mode = prim_rew_mode              # 'close' or 'touch' determining, whether rewards are given in case of touching or closeness
        self.rew_cutoff = rew_cutoff                    # for primitive rewards: the max distance to the rod that still gets rewarded
        self.start_conf = start_conf                    # which configuration to start with
        self.WLU_prefact = WLU_prefact                  # prefactor for the differential reward
        self.trans_dist = trans_dist                    # distance over which to transpport the rod in mode 7 (transportation)
        self.WLU_mode = WLU_mode                        # non-existing ('non_ex') or 'passive' particles in det_hypPerformance
        self.n_ep = train_ep                            # is needed in WLU_mode == 'switch'
        self.WLU_noise = WLU_noise                      # noise in determination of performance and hypPerformance for diff Rews
        self.sparse_rew = sparse_rew                    # gives only one, random particle a reward every step
        self.last_rew_part = []                         # needed for first step in sparse_rew
        self.n_rew_frames = n_rew_frames                # number of frames a particle is rewarded in the sparse_rew==true case
        self.lost = []                                  # no particles lost as default

        # parameters of dynamics
        self.int_steps = int_steps                      # number of integration steps done in every simulation step
        self.dt = dt
        self.Dt = Dt # 0.014
        self.Dr = Dr # 1.0 / 350.0
        self.Rm = math.sqrt(2*self.Dt/self.dt)
        self.Rr = math.sqrt(2*self.Dr/self.dt)
        self.vel_act = vel_act
        self.vel_tor = vel_tor
        self.torque = 1.0 / 350.0 * torque              # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)

        # type of task.
        # determines reward function, and observation space.
        # 1 - move rod
        # 2 - move rod along -x direction
        # 3 - rotate rod
        self.rewards = np.zeros(N)
        self.mode = mode
        if (self.mode == 2): #directional pushing
            self.n_obs += 2
        elif (self.mode == 4): #rotation with direction s
            self.n_obs += 1
        elif (self.mode == 6): #push along long direction
            self.n_obs += 1
        elif (self.mode == 7): #transport rod to target
            self.n_obs += 5
            self.start_conf = 'transportation'
            self.rew_mode = 'WLU'

        # target is always initialized to have something for the arguments in get_o_r
        self.target = np.zeros((self.n_rod, 2))

        if self.start_conf == 'standard':
            self.particles, self.rod = self.reinitialize_random_for_MD(swirl)
        elif self.start_conf == 'test_friction':
            self.particles, self.rod = self.reinitialize_test_friction()
        elif self.start_conf == 'biased':
            self.particles, self.rod = self.reinitialize_biased()
        elif self.start_conf == 'transportation':
            # If self.mode == 7, reinitialize_random_for_MD also gives a target position
            self.particles, self.rod, self.target = self.reinitialize_random_for_MD(swirl)

        # Old rod and particles are needed for evaluating the rod movement and the partice performance
        self.old_rod = np.zeros(self.rod.shape)
        self.old_rod[:] = self.rod[:]
        self.old_part = np.zeros(self.particles.shape)
        self.old_part = self.particles
        self.actions = np.zeros(self.particles.shape[0])
        self.old_actions = np.zeros(self.particles.shape[0])
        self.part_perf = np.zeros((self.particles.shape[0], 1))
        self.part_rod_forces = np.zeros((self.particles.shape[0], 3))
        self.rod_dist = np.zeros((self.particles.size))


    def update(self, particles, old_actions, rod, lost, update):
        '''
        This is for communicating with the experiment. It mainly
        deals with lost and found particles.
        '''

        # Deal with new (found) particles: since they do not have an old position,
        # they get a zero reward the first time they appear. That's why
        # they are not assigned to self.particles in the beginning
        # (found particles are always at the end)

        if update == 0:
            # In the very first update, only observables are calculated
            # and care must be taken to not have missmatching array shapes
            self.old_part = particles
            self.old_rod = rod

        # Found particles are always in the end
        found = np.full_like(lost, False)
        found[self.old_part.shape[0]:particles.shape[0]] = True

        self.rod = rod
        self.particles = particles[np.logical_and(~found, ~lost),:]

        # In the case of lost or particles, leave them out of the reward calculation
        self.old_part = self.old_part[~lost[~found],:]

        # Adjust old_actions according to old_part
        self.old_actions = old_actions[np.logical_and(~found, ~lost)]

        # The number of particles has to be adjusted every time, too (for fortran)
        self.N = sum(np.logical_and(~found, ~lost))

        # obs and rewards have to be preallocated, because they are longer than get_obs_rewards' output
        rewards = np.full_like(particles[:,0], np.nan)
        obs = np.full((particles.shape[0], 10), np.nan)

        # Compute observables and rewards from particles and rod
        obs[np.logical_and(~found, ~lost), :], rewards[np.logical_and(~found, ~lost)] = self.get_obs_rewards()
        obs[found, :] = 0
        rewards[found] = 0

        # In the first update, rewards are zero, since there are no old positions, etc.
        if update == 0:
            rewards[np.logical_and(~found, ~lost)] = 0

        # Update the environment (found particles are included now)
        self.old_rod = rod
        self.old_part = particles[~lost,:]

        return obs, rewards, found


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
            l_rod = self.n_rod*self.dist_rod
            particles = np.zeros((self.N,3))
            particles[:self.N//2,2] = np.pi/2
            particles[self.N//2:,2] = -np.pi/2
            particles[:self.N//2,0] = 1
            particles[self.N//2:,0] = -1
            particles[:self.N//2,1] = np.arange(-l_rod/2, l_rod/2, l_rod*2/(self.N-1))
            particles[self.N//2:,1] = np.arange(-l_rod/2, l_rod/2, l_rod*2/(self.N-1))

        particles[particles[:,0] <= 0] -= [10.0, 0, 0]
        particles[particles[:,0] > 0] += [10.0, 0, 0]


        rod = np.array([[0.0, (i-(self.n_rod-1)/2)*self.dist_rod] for i in np.arange(self.n_rod)])

        if self.mode == 7:
            # Adds a target position in the form of another rod at a
            # random angle and a distance self.trans_dist from the origin

            target = self.make_target()

            return particles, rod, target

        return particles, rod

  # Initialize with two particles close to the rod and at one end of it (for test_friction)

    def reinitialize_test_friction(self):
        # Initializes with two particles close to the rod at one of its ends
        particles = np.array([[-7, -40, np.pi/2-np.pi/4], [7, -40, np.pi/2+np.pi/4]])

        rod = np.array([[0.0, (i-(self.n_rod-1)/2)*self.dist_rod] for i in np.arange(self.n_rod)])

        return particles, rod


    def reinitialize_biased(self):
        # Initializes with the partielces on opposite sides at the ends of the rod and pointing towards it
        grid_sz = np.int(np.sqrt(self.N/2)) + 1
        grid_vals = [10 * (element + 0.5 - grid_sz/2) for element in list(range(grid_sz))]

        grid0 = np.array(grid_vals, ndmin=2) + 1j * np.transpose(np.array(grid_vals, ndmin=2)) # Prototype of the grid
        grid1 = grid0 + (10 * (1  + grid_sz/2) + 1j * (self.l_rod/2 - 10 * grid_sz/2)) # Grid shifted to the upper right end of the rod
        grid2 = grid0 + (-10 * (1  + grid_sz/2) - 1j * (self.l_rod/2 - 10 * grid_sz/2)) # Grid shifted to the lower left end of the rod

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

        rod = np.array([[0.0, (i-(self.n_rod-1)/2)*self.dist_rod] for i in np.arange(self.n_rod)])

        return particles, rod


    def make_target(self):
        '''
        This adds a target rod random angle and distance self.trans_dist to the origin
        '''
        # Two random angles in the intervall [-pi; pi] for the direction and the orientation of the rod
        pos_angle = 2 * np.pi * np.random.rand(1) - np.pi
        or_angle = 2 * np.pi * np.random.rand(1) - np.pi

        # making the position of the target complex
        cm_comp = self.trans_dist * e ** (1j * pos_angle)
        target_ends = np.array([cm_comp + self.l_rod/2 * e ** (1j*or_angle), cm_comp - self.l_rod/2 * e ** (1j*or_angle)])
        target_comp = np.linspace(target_ends[0], target_ends[1], self.n_rod)

        # making the position of the target real
        target = np.zeros((self.n_rod, 2))
        target[:,0] = np.real(target_comp).transpose()
        target[:,1] = np.imag(target_comp).transpose()

        return target


    def evolve_MD(self, actions, rot_dir=0, old_rot_dir=0):
        '''
        Evolves the particle and rod situation one simulation step (=int_steps times one dt time-step)
        '''

        X = self.particles[:,0]
        Y = self.particles[:,1]
        T = self.particles[:,2]
        x_rod = self.rod[:,0]
        y_rod = self.rod[:,1]
        fr_rod = self.fr_rod
        inert_rod = self.inert_rod

        # Assign the old quantities
        self.old_rod[:] = self.rod[:]
        self.old_part[:] = self.particles[:]
        self.old_actions[:] = actions[:]

        # these are necessary due to the possibility to reproduce a step with the old noise and reproduction == True
        reproduction = False
        old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
        old_vel_noise = np.zeros((self.N, self.int_steps))
        old_tor_noise = np.zeros((self.N, self.int_steps))

        # Is there noise at all?
        # WLU_noise == 'no' means no noise at all, also not in the real simulation steps
        if self.WLU_noise == 'no':
            noise_flag = 0
        else:
            noise_flag = 1

        self.particles, self.rod, self.part_rod_forces, \
        self.old_ther_noise, self.old_vel_noise, self.old_tor_noise = evolve.evolve_md_rod(fr_rod, inert_rod,
                                    X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                    x_rod, y_rod, self.dist_rod, actions,
                                    self.Rm, self.Rr, self.dt,
                                    self.torque, self.vel_act, self.vel_tor,
                                    self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                    noise_flag, self.N, self.n_rod, self.int_steps)

        obs, rewards = self.get_obs_rewards(rot_dir, old_rot_dir)

        # Calculating the rod orientation and CoM for saving it to the stats file
        rod_theta = np.angle(complex(self.rod[-1,0] - self.rod[0,0], self.rod[-1,1] - self.rod[0,1]))

        rod_cm = np.zeros((1,1,2)) # rod_cm[0,0,0] is x-component and rod_cm[0,0,1] is y-component
        rod_cm[0,0,0] = np.mean(self.rod[:,0])
        rod_cm[0,0,1] = np.mean(self.rod[:,1])

        return obs, rewards, rod_theta, rod_cm



  # CALLS THE FORTRAN SUBROUTINE FOR OBS AND REWARDS IN PRESENCE OF A ROD
    def get_obs_rewards(self, rot_dir=0, old_rot_dir=0, flag_side=0, obs_type=1):

        # Determining the performance P of each particle (this is the important part)
        p = self.particles
        r = self.rod
        olr = self.old_rod
        tar = self.target
        if (self.mode == 4):
            assert rot_dir in [-1,1]
            assert old_rot_dir in [-1,1]

        # Now the observables are determined, if rew_mode=='classic' the rewards determined here are used, too
        obs, rew_classic, self.touch, self.rod_dist = evolve.get_o_r_rod(p[:,0], p[:,1], p[:,2],
                                          r[:,0], r[:,1], olr[:,0], olr[:,1], tar[:,0], tar[:,1],
                                          self.mode, rot_dir, old_rot_dir,
                                          flag_side, self.flag_LOS,
                                          self.part_size, self.part_size_rod, self.fr_rod,
                                          self.ext_rod, self.cen_rod,
                                          obs_type,
                                          self.cones, self.cone_angle, self.close_pen, self.prox_rew, self.flag_fix_or,
                                          self.n_obs, self.N, self.n_rod)

        if self.rew_mode == 'classic':

            # Rewards based on position along and orientation with respect to the rod
            rewards = rew_classic

        elif self.rew_mode == 'forces':

            # Determines the rewards according to the forces and the current mode
            rewards = self.get_forces_rewards(flag_abs=False)

        elif self.rew_mode == 'abs_forces':

            # Determines the rewards according to the forces and te current mode
            rewards = self.get_forces_rewards(flag_abs=True)

        elif self.rew_mode == 'primitive' or self.rew_mode == 'approx_diff':

            # Determines rewards in primitive way (close? rotated?) So far only for rot.
            # the prim_rew_mode specifies if touching or closeness are decisive.
            # In the case of approx_diff, an reward estimation during passive actions
            # is subtracted in learning_rod for approximating difference rewards.
            rewards = self.get_primitive_rewards(prim_rew_mode=self.prim_rew_mode)

        elif self.rew_mode == 'WLU':

            # Determines the reward according to what would have happened if particle i would not have been there.
            # (Wonderful Life Utility, WLU)
            rewards = self.get_WLU()

        elif self.rew_mode == 'WLU_experiment':

            # Determines the reward according to what would have happened if particle i would not have been there.
            # (Wonderful Life Utility, WLU) It also uses a scaling that makes the experiment and the simulations
            # more compatible
            rewards = self.get_WLU_experiment()

        # Gives only one, random particle a reward every step
        if self.sparse_rew:

            # selecting one particle that is rewarded in the next n_rew_frames frames
            rand_part = random.randrange(self.N)

            # update last_rew_part (all particles that are not lost this frame)
            self.last_rew_part.extend([num for num in set(self.last_rew_part)])
            self.last_rew_part.append(rand_part)

            # the particle is non-lost for self.nFramesLost + 1 frames, then it is lost again
            # toRemove = [num for num in self.last_rew_part if self.last_rew_part.count(num) > self.n_rew_frames + 1]

            # if toRemove: for num in toRemove: self.last_rew_part.remove(num)

            self.last_rew_part = list(num for num in self.last_rew_part if self.last_rew_part.count(num) < self.n_rew_frames + 1)

            # give no reward to the other particles ...
            rewards[[part_num not in self.last_rew_part for part_num in range(self.N)]] = 0

            # ... and make them all lost
            self.lost = np.unique(self.last_rew_part)

        self.rewards = rewards

        return obs, rewards


    def get_forces_rewards(self, flag_abs=False):
        '''
        This calculates a reward based on the contribution of each particle to the performance.
        The contribution of each particle is estimated by evaluating how well the forces the
        particle exerted on the rod meet the desired outcome.
        '''
        r = self.rod
        olr = self.old_rod

        performance = self.det_performance(self.rod)

        if self.mode == 3: # Rotation
            d_theta_uncorr = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1])) - \
                np.angle(complex(olr[-1,0] - olr[0,0], olr[-1,1] - olr[0,1])) # Still can have jumps

            d_theta = d_theta_uncorr - np.floor(d_theta_uncorr/(2 * np.pi) + 0.5) * 2 * np.pi # Now the jumps are corrected

            self.part_perf = self.part_rod_forces[:,2] * np.sign(d_theta) # Performance is proportional torque fr rotation

        elif self.mode == 6: # Longitudinal pushing

            rod_theta = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1]))

            # When rew_mode is 'forces' (not 'abs_forces'), the particle performance are
            # the forces exerted on the rod in the right direction
            if not flag_abs:

                part_rod_forces_complex = self.part_rod_forces[:,0] + 1j * self.part_rod_forces[:,1]

                self.part_perf = (part_rod_forces_complex * e ** (-1j*rod_theta)).real # Particle forces in the longitud. direction of the rod

            # When rew_mode is 'abs_forces' (meaning flag_abs=True), the particle
            # performance is just the sum of the absolute forces a particle exerts on
            # the rod, meaning it is rewarded more, if it is interacting more.
            elif flag_abs:

                self.part_perf = np.sum(abs(self.part_rod_forces), axis=1)


        rewards = self.p_rew_fact * performance * abs(self.part_perf) # Performance is only rewarded, if the rod has moved


        return rewards


    def get_primitive_rewards(self, prim_rew_mode='touch'): # the prim_rew_mode spwcifies if touching or closeness are decisive
        '''
        This simply rewards every particle that is present within a certain
        area around the rod if the rod has moved or rotated, etc.
        '''

        performance = self.det_performance(self.rod)

        if self.mode == 3: # Rotation
            ref_prefactor = self.r_rew_fact
        elif self.mode == 6: # Long. Trans.
            ref_prefactor = self.p_rew_fact

        if prim_rew_mode == 'close':
            r = self.rod
            p = self.particles

            # Determining the distances to the rod (for every mode)
            p_comp = np.array(p[:,0] + 1j * p[:,1], ndmin=2)
            r_comp = np.array(r[:,0] + 1j * r[:,1], ndmin=2)

            dist = np.transpose(abs(p_comp - np.transpose(r_comp))) # Particles are in rows with their distances to the rod in columns
            min_dist = np.transpose(np.amin(dist, axis=1))
            close_enough = min_dist <= self.rew_cutoff # Only the particles within the cutoff distance to the rod get rewarded

            rewards = close_enough * performance * ref_prefactor # The direction of rotation does not matter.

        elif prim_rew_mode == 'touch':

            rewards = self.touch * performance * ref_prefactor # The direction of rotation does not matter.

        return rewards


    def get_WLU(self):
        '''
        Determines the reward for particle i according to how the performance would have
        changed if particle i would not have been present (hypPerformance).
        This is as general as possible, while it is very simple,
        since all forces are considered automaticaly.
        Yes, this is computationally very expensive.
        '''

        # In the initialization, determining this type of reward is not possible
        if not sum(self.old_actions):
            rewards = np.zeros(self.particles.shape[0])

            # For saving the hypothetical particle positions I need an N_rod x rod.shape[1] x N array
            hyp_rod = np.zeros((self.rod.shape[0], self.rod.shape[1], self.particles.shape[0]))
            # For saving the hypothetical particle positions I need an N x particles.shape[1] x N array
            hyp_parts = np.zeros((self.old_part.shape[0], self.old_part.shape[1], self.particles.shape[0]))

            # This is needed for having a consistent output from .update
            self.hyp_rod = hyp_rod
            self.hyp_parts = hyp_parts

            return rewards

        # First up, the performance is determined according to the mode (translation, rotation, etc.)
        # for the case without noise (WLU_noise = 'off') this is done from a noislessRod determined
        # from the last simulation step without noise
        if self.WLU_noise == 'on' or self.WLU_noise == 'mixed' or self.WLU_noise == 'ideal' or self.WLU_noise == 'no':

            perf_rod = self.rod

        elif self.WLU_noise == 'off':
            # redo the last simulation step without noise to have a performance
            # that can be compared to the hyp_perf without noise

            noise_flag = 0

            X = self.old_part[:,0]
            Y = self.old_part[:,1]
            T = self.old_part[:,2]
            action = self.old_actions[:]
            N = self.N
            x_rod = self.old_rod[:,0]
            y_rod = self.old_rod[:,1]
            fr_rod = self.fr_rod
            inert_rod = self.inert_rod

            # these are necessary due to the possibility to reproduce the last step with the old noise and reproduction == True
            reproduction = False
            old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
            old_vel_noise = np.zeros((self.N, self.int_steps))
            old_tor_noise = np.zeros((self.N, self.int_steps))

            _, perf_rod, _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                                X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                                x_rod, y_rod, self.dist_rod, action,
                                                self.Rm, self.Rr, self.dt,
                                                self.torque, self.vel_act, self.vel_tor,
                                                self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                                noise_flag, N, self.n_rod, self.int_steps)

        performance = self.det_performance(perf_rod)

        # The hyp_perf are the hypothetical performances that would have been achieved in the absence of particle i
        hyp_perf, hyp_rod_ang, hyp_rod, hyp_parts = self.det_hyp_perf(performance)

        # The contribution of a particle is the difference between the actual performance
        # and the hypothetical performance if it would not have been there.
        contrib = (performance - hyp_perf)

        # Really with performance here? Yes, because otherwise opposing particles get both rewarded even though nothing happens.
        # Wolpert and Tumer (2001) do not multiply the performance here.
        # In the case of the transportatioon problem, the rewards ar cummulated so they
        # rise from 0 to a constant value upon completion of the task
        if self.mode == 7:
            rewards = self.rewards + contrib
        else:
            rewards = self.WLU_prefact * contrib

        # For debugging the performance and hyp_perf are saved together with the rod
        # the performance was determined from and the hypothetical rods (just angles in for lattter two)

        perf_rod_ang = np.angle(complex(perf_rod[-1,0] - perf_rod[0,0], perf_rod[-1,1] - perf_rod[0,1]))

        self.hyp_rod_ang = hyp_rod_ang
        self.hyp_perf = hyp_perf
        self.performance = performance
        self.perf_rod_ang = perf_rod_ang
        self.hyp_rod = hyp_rod
        self.hyp_parts = hyp_parts

        return rewards


    def get_WLU_experiment(self):
        '''
        Determines the reward for particle i according to how the performance would have
        changed if particle i would not have been present (hypPerformance).
        This is as general as possible, while it is very simple,
        since all forces are considered automaticaly.
        Yes, this is computationally very expensive.
        This version of it enforces the mixed noise mode
        '''

        # Because that's the only working noise mode for the experiment
        self.WLU_noise = 'mixed'

        # In the initialization, determining this type of reward is not possible
        if not sum(self.old_actions):
            rewards = np.zeros(self.particles.shape[0])

            # For saving the hypothetical particle positions I need an N_rod x rod.shape[1] x N array
            hyp_rod = np.zeros((self.rod.shape[0], self.rod.shape[1], self.particles.shape[0]))
            # For saving the hypothetical particle positions I need an N x particles.shape[1] x N array
            hyp_parts = np.zeros((self.old_part.shape[0], self.old_part.shape[1], self.particles.shape[0]))

            # This is needed for having a consistent output from .update
            self.hyp_rod = hyp_rod
            self.hyp_parts = hyp_parts

            return rewards

        # First up, the performance is determined according to the mode (translation, rotation, etc.)

        # There are two different rods now: the experimental and the virtual one.
        # We need both for scaling the rotations in the resimulation steps
        experiment_rod = self.rod

        # Get the virtual rod
        # Redo the last simulation step without noise to get the virtual rod (without noise)
        noise_flag = 0

        X = self.old_part[:,0]
        Y = self.old_part[:,1]
        T = self.old_part[:,2]
        action = self.old_actions[:]
        N = self.N
        x_rod = self.old_rod[:,0]
        y_rod = self.old_rod[:,1]
        fr_rod = self.fr_rod
        inert_rod = self.inert_rod

        # these are necessary due to the possibility to reproduce the last step with the old noise and reproduction == True
        reproduction = False
        old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
        old_vel_noise = np.zeros((self.N, self.int_steps))
        old_tor_noise = np.zeros((self.N, self.int_steps))

        _, virtual_rod, _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                            X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                            x_rod, y_rod, self.dist_rod, action,
                                            self.Rm, self.Rr, self.dt,
                                            self.torque, self.vel_act, self.vel_tor,
                                            self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                            noise_flag, N, self.n_rod, self.int_steps)

        virtual_performance = self.det_performance(virtual_rod)
        experiment_performance = self.det_performance(experiment_rod)

        # The hyp_perf are the hypothetical performances that would have been achieved in the absence of particle i
        # the experiment performance is given here, because that is the baseline
        hyp_perf, hyp_rod_ang, hyp_rod, hyp_parts = self.det_hyp_perf(virtual_performance)

        # The contribution of a particle is the difference between the actual performance
        # and the hypothetical performance if it would not have been there,
        # scaled such that if the effect of this particle tends to 0, the performance
        # matches the experimental performance
        if virtual_performance == 0 or np.any(abs(hyp_perf/virtual_performance) >= 10):
            contrib = experiment_performance - hyp_perf
        else:
            contrib = experiment_performance - hyp_perf * experiment_performance/virtual_performance

        # Wolpert and Tumer (2001) do not multiply the performance here.
        rewards = self.WLU_prefact * contrib # * performance

        # For debugging the performance and hyp_perf are saved together with the rod
        # the performance was determined from and the hypothetical rods (just angles in for latter two)
        perf_rod_ang = np.angle(complex(virtual_rod[-1,0] - virtual_rod[0,0], virtual_rod[-1,1] - virtual_rod[0,1]))

        self.hyp_rod_ang = hyp_rod_ang
        self.hyp_perf = hyp_perf
        self.performance = experiment_performance
        self.perf_rod_ang = perf_rod_ang
        self.hyp_rod = hyp_rod
        self.hyp_parts = hyp_parts

        return rewards


    def det_performance(self, rod):
        '''
        This determines the performance (e.g. how far the rod rotated in the last step) according to the mode.
        '''

        r = rod
        olr = self.old_rod
        t = self.target

        rod_theta = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1]))


        if self.mode == 3: # Rotation

            d_theta_uncorr = rod_theta - np.angle(complex(olr[-1,0] - olr[0,0], olr[-1,1] - olr[0,1])) # Still can have jumps

            d_theta = d_theta_uncorr - np.floor(d_theta_uncorr/(2 * np.pi) + 0.5) * 2 * np.pi # Now the jumps are corrected

            performance = abs(d_theta)

        elif self.mode == 6: # Longitudinal transport
            # The performance is determined by the product of the CoM motion of the rod (d_cm)
            # projected on both the new and old rod's long axis. This is done to ensure, that
            # if the rod's orientation has changed, the reward is smaller. (see Labbook 08.06.21)

            d_cm = complex(sum(r[:,0]) / self.n_rod - sum(olr[:,0]) / self.n_rod, \
                sum(r[:,1]) / self.n_rod - sum(olr[:,1]) / self.n_rod) # Center of mass motion in complex numbers

            old_rod_theta = np.angle(complex(olr[-1,0] - olr[0,0], olr[-1,1] - olr[0,1]))

            d_cm_lon_new = abs((d_cm * e ** (-1j*rod_theta)).real) # Motion projected on the new rod axis

            d_cm_lon_old = abs((d_cm * e ** (-1j*old_rod_theta)).real) # Motion projected on the old rod axis

            performance = d_cm_lon_new * d_cm_lon_old

        elif self.mode == 7: # Transportation problem
            # The "values" of a certain rod position is determined by
            # sum_i (1/(d_i + 1))
            # with all particles i and the distance to the corresponding target particle d_i.
            # The perfrmance is then determined by the change in the "value od the rod.

            # complex representations of everything (old and new rod and target)
            tar_c = t[:,0] + 1j *  t[:,1]
            rod_c = r[:,0] + 1j *  r[:,1]
            # olr_c = olr[:,0] + 1j *  olr[:,1]

            # determining the distances
            dists_new = abs(tar_c - rod_c)
            # dists_old = abs(tar_c - olr_c)

            # The value is linearly decreasing from 100 to 0 over
            # the distance (from 0 to its initial value):
            value_new = 100 * (1 - np.mean(dists_new) / self.trans_dist)
            # value_old = 100 * (1 - np.mean(dists_old) / self.trans_dist)

            # determining the performance from the change in the value
            # (without subtracting the old value, since this would give a reward of 0 when the particles have achieved their goal)
            performance = value_new # - value_old

        return performance


    def det_hyp_perf(self, performance):
        '''
        This determines for every particle, how the performance would have been in
        the absence of this particle. It needs the old particle and rod positions
        as well as the old actions.
        For every particle, it evolves the environment one step without this particle.
        '''

        # Set the noise for determining the hypothetical performances:
        # 'on':    noise in determining performance and hypPerf
        # 'off':   no noise in determining perf and hypPerf
        # 'mixed': noise in determining perf and no noise in determining hypPerf
        # 'ideal': exactly the same noise in both perf and hypPerf calculations

        if self.WLU_noise == 'on':
            noise_flag = 1
        elif self.WLU_noise == 'off' or self.WLU_noise == 'mixed' or self.WLU_noise == 'ideal' or self.WLU_noise == 'no':
            noise_flag = 0


        # in the WLU_noise 'ideal', the same noise as in the last step is used to determine the hypPerfs
        if self.WLU_noise == 'ideal':
            reproduction = True
            old_ther_noise = self.old_ther_noise
            old_vel_noise = self.old_vel_noise
            old_tor_noise = self.old_tor_noise
        else:
            reproduction = False
            old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
            old_vel_noise = np.zeros((self.N, self.int_steps))
            old_tor_noise = np.zeros((self.N, self.int_steps))

        hyp_perf = np.zeros(self.particles.shape[0])
        hyp_rod_ang = np.zeros(self.particles.shape[0]) # this is just the angle
        # For saving the hypothetical particle positions I need an N_rod x rod.shape[1] x N array
        hyp_rod = np.zeros((self.rod.shape[0], self.rod.shape[1], self.particles.shape[0]))
        # For saving the hypothetical particle positions I need an N x particles.shape[1] x N array
        hyp_parts = np.zeros((self.old_part.shape[0], self.old_part.shape[1], self.particles.shape[0]))

        # Define the starting configuretion
        x_rod = self.old_rod[:,0]
        y_rod = self.old_rod[:,1]
        fr_rod = self.fr_rod
        inert_rod = self.inert_rod

        distances = self.rod_dist
        touch = self.touch

        # Iterate over every particle, leave out that particle (WLU_mode = 'non_ex')
        # or make it passive (WLU_mode = 'passive') and simulate one step.
        for i in range(self.particles.shape[0]):

            if (distances[i] <= self.rew_cutoff) or touch[i] :

                if self.WLU_mode == 'non_ex':
                    # Make particle i non-existing

                    # Leave out particle i
                    mask = np.ones(self.old_part.shape[0], dtype=bool)
                    mask[i] = 0

                    X = self.old_part[mask,0]
                    Y = self.old_part[mask,1]
                    T = self.old_part[mask,2]
                    action = self.old_actions[mask]

                    # All the old noises are masked, too (diffusion, velocity and torque)
                    old_th_n = old_ther_noise[mask, :]
                    old_v_n = old_vel_noise[mask, :]
                    old_tor_n = old_tor_noise[mask, :]

                    N = self.N - 1 # Don't forget the particle number

                elif self.WLU_mode == 'passive':
                    # Make particle i passive

                    X = self.old_part[:,0]
                    Y = self.old_part[:,1]
                    T = self.old_part[:,2]
                    action = self.old_actions[:]
                    N = self.N

                    action[i] = 0

                    old_th_n = old_ther_noise
                    old_v_n = old_vel_noise
                    old_tor_n = old_tor_noise

                    # So the equivalent of the mask is all ones
                    mask = np.ones(self.old_part.shape[0], dtype=bool)

                hyp_parts[mask,:,i], hyp_rod[:,:,i], _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                                X, Y, T, old_th_n, old_v_n, old_tor_n,
                                                x_rod, y_rod, self.dist_rod, action,
                                                self.Rm, self.Rr, self.dt,
                                                self.torque, self.vel_act, self.vel_tor,
                                                self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                                noise_flag, N, self.n_rod, self.int_steps)

                if self.WLU_mode == "non_ex":
                    hyp_parts[i,:,i] = self.old_part[i,:]

                # Now the hypPerformance in the absence of particle i is determined
                hyp_perf[i] = self.det_performance(hyp_rod[:,:,i])

            else:
                # If the particle is too far away to have an effect (distance > rew_cutoff),
                # the performance without this particle should be the same as with it.
                hyp_perf[i] = performance
                # the hypothetical particles and rod must still be written
                hyp_rod[:,:,i] = self.rod
                hyp_parts[:,:,i] = self.old_part

            # for debugging, the hypothetical rod angles are also returned
            hyp_rod_ang[i] = np.angle(complex(hyp_rod[-1,0,i] - hyp_rod[0,0,i], hyp_rod[-1,1,i] - hyp_rod[0,1,i]))

        return hyp_perf, hyp_rod_ang, hyp_rod, hyp_parts

#
# ------------------------------------
# End of class MD
# ------------------------------------
