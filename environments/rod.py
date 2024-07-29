# ---------------------------------------
import numpy as np
import math
from math import e
from multiprocessing import Pool
import random
from fortran import evolve_environment as evolve
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
    def __init__(self, N=30, size=10, skew=False,
                int_steps=20, vel_act=0.35, vel_tor=0.2, dt=0.2, torque=25.0,
                vel_noise_fact=0.5, rot_noise_fact=0.5,
                fr_rod=10., inert_rod=1., l_rod=100, n_rod=60, ext_rod=1.0, cen_rod=1.0, mu_K=0.0,
                Dt=0.014, Dr=1.0 / 350.0,
                obs_type='1overR', cones=5, n_obs=5, cone_angle=np.pi, flag_side=False, flag_LOS=False,
                part_size=6.2, part_size_rod=0.0, mode=1, swirl=False,
                data_path=None, rew_mode='CR', team_rew_mode='close', CR_mode = 'non_ex',
                close_pen=0, prox_rew=0, r_rew_fact=2, CR_prefact=10000, CR_noise='mixed', CR_rew_mode='close',
                rew_cutoff=60, start_conf='standard', start_dist_scale=1, trans_dist=100, target_tol=120, final_rew=1000, cost_iso_rew=False,
                CR_touch_rew=0.1, termination_mode="ind", achieved_dist=6,
                flag_fix_or = 0, train_ep = 100, n_processes=1, parallelize_cr=False,
                use_obst=False, obst_conf='random', obst_vision=False,
                **unused_parameters):

        # The task is always not achieved in the beginning
        # (whether or not it can be achieved)
        self.task_achieved = False

        # path for writing the trajectories
        self.data_path = data_path
        # internal knowledge of system
        self.skew = skew
        self.N = N
        self.size = size

        # Basic objects
        self.particles = np.zeros((N,3))
        self.rod = np.zeros((n_rod,2))
        self.obstacles = np.zeros((1,2))
        self.n_obst = 1

        # sight characterization:
        # obs_type -> 1/r, 1/r^2
        # cones -> number of cones
        # cone angle -> [-cone_angle/2, cone_angle/2] sight, in deg, converted in rad
        # flag_side -> whether particles can see through the rod!
        assert obs_type in ['1overR', '1overR2'], 'Obs Type not Recognized'
        if obs_type=='1overR':
            self.obs_type = 1
        elif obs_type=='1overR2':
            self.obs_type = 2
        self.cones = cones
        self.n_obs = n_obs
        self.cone_angle = cone_angle
        self.flag_side = int(flag_side)
        self.flag_LOS = int(flag_LOS)
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
        self.final_rew = final_rew                      # The reward upon achieved task for truely episodic learning
        self.cost_iso_rew = cost_iso_rew                # Cost instead of reward in episodic task mode 7
        self.termination_mode = termination_mode
        self.achieved_dist = achieved_dist

        self.r_rew_fact = r_rew_fact                    # These are factors for the implementation of rewards based on forces
        self.rew_mode = rew_mode                        # 'CR', 'forces' or 'torque'
        self.team_rew_mode = team_rew_mode              # 'team', 'close' or 'touch' determining, whether rewards are given in case of touching or closeness
        self.rew_cutoff = rew_cutoff                    # for team rewards: the max distance to the rod that still gets rewarded
        self.start_conf = start_conf                    # which configuration to start with
        self.start_dist_scale = start_dist_scale        # scaling factor for the starting positions of the particles to initialize them far away
        self.CR_prefact = CR_prefact                    # prefactor for the differential reward
        self.trans_dist = trans_dist                    # distance over which to transpport the rod in mode 7 (transportation)
        self.target_tol = target_tol                    # allowed residual cummulative distance between target and rod for completion of the task
        self.CR_mode = CR_mode                          # non-existing ('non_ex') or 'passive' particles in det_hypPerformance
        self.n_ep = train_ep                            # is needed in CR_mode == 'switch'
        self.CR_noise = CR_noise                        # noise in determination of performance and hypPerformance for diff Rews
        self.CR_rew_mode = CR_rew_mode                  # which particles are considered for CR
        self.CR_touch_rew = CR_touch_rew                # Rewards for touching in case of CR
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
        self.vel_noise_fact = vel_noise_fact
        self.rot_noise_fact = rot_noise_fact
        self.torque = 1.0 / 350.0 * torque              # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)

        # type of task.
        # determines reward function, and observation space.
        # 1 - move rod
        # 2 - move rod along -x direction
        # 3 - rotate rod
        self.rewards = np.zeros(N)
        self.mode = mode

        # Concerning obstacles
        self.use_obst = use_obst       # whether or not to include obstacles
        self.obst_conf = obst_conf     # configuration of obstacles ('random' or 'wall')
        self.obst_vision = obst_vision # whether or not the robots can see the obstacles

        # target is always initialized to have something for the arguments in get_o_r
        self.target = np.zeros((self.n_rod, 2))

        if self.start_conf == 'standard':
            self.reinitialize_random_for_MD(swirl)
        elif self.start_conf == 'test_friction':
            self.reinitialize_test_friction()
        elif self.start_conf == 'biased':
            self.reinitialize_biased()
        elif self.mode == 7:
            # If self.mode == 7, reinitialize_random_for_MD also gives a target position
            # Transversal transportation -> target parallel and in orthogonal direction to rod
            # Transversal transportation -> target parallel and in orthogonal direction to rod
            self.reinitialize_random_for_MD(swirl)

        # If there is only one particle, CR reward gets changed to
        # team rewards
        if self.N == 1 and self.rew_mode =="CR":
            self.rew_mode = "team"
            self.team_rew_mode = "team"

        # Scale the particle positions in order to be able to initialize them far away
        self.particles[:,0:2] = self.start_dist_scale * self.particles[:,0:2]

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
        self.ids = np.full_like(self.actions, np.nan)
        self.old_ids = np.full_like(self.actions, np.nan)
        self.virtual_performance = 0

        # Parameters for parallelization
        self.parallelize_cr = parallelize_cr
        self.n_processes = n_processes


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
        obs = np.full((particles.shape[0], self.n_obs), np.nan)

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


    def update_no_training(self, particles, rod, update):
        '''
        This is for communicating with the experiment.
        In this version of 'update', there are only present particles
        and no training with counterfactuals is possible, which
        makes it simpler to only execute models.
        '''

        # Keep the old rod orientation for rewards
        if update == 0:
            self.old_rod = rod

        # The present particles are simply all particles that were received
        self.particles = particles
        self.rod = rod
        self.N = particles.shape[0]

        # We only need observables here so we set the rewarding mode to the
        # most simple scheme
        self.rew_mode = 'torque'
        obs, rewards = self.get_obs_rewards()

        # Update the old rod
        self.old_rod = self.rod

        return obs, rewards


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

        # Assign rod and particles to the environment
        self.particles = particles
        self.rod = np.array([[0.0, (i-(self.n_rod-1)/2)*self.dist_rod] for i in np.arange(self.n_rod)])

        if self.mode == 7:
            # Adds a target position in the form of another rod at a
            # random angle and a distance self.trans_dist from the origin
            self.make_target()

        if self.use_obst:
            # Add obstacles
            self.add_obstacles()

        return


    def reinitialize_test_friction(self):
        '''
        Initializes with two particles close to the rod at one of its ends
        '''
        self.particles = np.array([[-7, -40, np.pi/2-np.pi/4], [7, -40, np.pi/2+np.pi/4]])

        self.rod = np.array([[0.0, (i-(self.n_rod-1)/2)*self.dist_rod] for i in np.arange(self.n_rod)])

        return


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

        # Assign everything to the environment
        self.rod = np.array([[0.0, (i-(self.n_rod-1)/2)*self.dist_rod] for i in np.arange(self.n_rod)])
        self.particles = particles

        return


    def make_target(self):
        '''
        This adds a target at a position and orientation depending on start_conf
        '''

        # If we use obstacles, let's only look at one scenario
        if self.use_obst:
            self.start_conf = 'transp_obstacles'

        # Get the complex rod first
        rod_comp = self.rod[:,0] + 1j * self.rod[:,1]

        if self.start_conf == 'transportation':
            # Two random angles in the intervall [-pi; pi] for the direction and the orientation of the rod
            pos_angle = 2 * np.pi * np.random.rand(1) - np.pi
            or_angle = 2 * np.pi * np.random.rand(1) - np.pi

            # making the position of the target complex
            cm_comp = self.trans_dist * e ** (1j * pos_angle)
            target_ends = np.array([cm_comp + self.l_rod/2 * e ** (1j*or_angle), cm_comp - self.l_rod/2 * e ** (1j*or_angle)])
            target_comp = np.linspace(target_ends[0], target_ends[1], self.n_rod)

        elif self.start_conf == 'transportation_long':
            # The target sits in longitudinal direction of the rod
            target_comp = rod_comp + self.trans_dist * e ** (1j * np.angle(rod_comp[-1] - rod_comp[0]))

        elif self.start_conf == 'transportation_trans':
            # The target sits in transversal direction of the rod
            target_comp = rod_comp + self.trans_dist * e ** (1j * (np.angle(rod_comp[-1] - rod_comp[0]) + np.pi/2))

        elif self.start_conf == 'transp_1':
            # The target is perpendicular to the rod an located to its side
            target_comp = rod_comp * e ** (1j * np.pi/2) + 1j * self.trans_dist

        elif self.start_conf == 'transp_2':
            # The target is perpendicular to the rod an located at its end
            target_comp = rod_comp * e ** (1j * np.pi/2) + self.trans_dist * e ** (1j * np.angle(rod_comp[-1] - rod_comp[0]))

        elif self.start_conf == 'transp_obstacles':
            # The target is perpendicular to the rod an located at its end
            target_comp = rod_comp * e ** (1j * np.pi/2) + self.trans_dist

        # making the position of the target real
        target = np.zeros((self.n_rod, 2))
        target[:,0] = np.real(target_comp).transpose()
        target[:,1] = np.imag(target_comp).transpose()

        # Assign the target to the environment
        self.target = target

        return


    def add_obstacles(self):
        '''
        Adds obstacles to the start configuration. Obstacles are immobile objects that can not be
        penetrated by particles or rod and are perceived by the particles
        '''
        # Chose the right obstacle configuration
        if self.obst_conf == "random":

            # Add a random configuration of obstacles
            self.obstacles = np.array([[40, 20], [30,-40], [90,-25], [55,75], [-40,30], [-30,-80], [50,-70]])

        elif self.obst_conf == "wall":

            # Add a wall between the particles and the rod (also move the particles)
            self.obstacles = self.rod.copy()
            self.obstacles[:,0] = self.obstacles[:,0] - 52
            self.particles[:,0] = self.particles[:,0] - 100

        elif self.obst_conf == "wall_close":

            # Add a wall between the particles and the rod (also move the particles)
            self.obstacles = self.rod.copy()
            self.obstacles[:,0] = self.obstacles[:,0] - 40
            self.particles[:,0] = self.particles[:,0] - 100

        elif self.obst_conf == "target_inside_tube":

            # Make a tube for obstacles
            self.obstacles = self.make_tube()

            # Put the target on the other side of the tube
            target = (self.rod.copy()[:,0] + 1j * self.rod.copy()[:,1]) * e ** (-1j * np.pi/2)
            self.target = np.stack((np.real(target), np.imag(target)), axis=1)

            # The tube with the rod and the particles on one side and the target on the other side
            self.rod[:,0] = self.rod[:,0] - 40
            self.particles[:,0] = self.particles[:,0] - 92

        elif self.obst_conf == "target_across_tube":

            # Make a tube for obstacles
            self.obstacles = self.make_tube()

            # Put the target on the other side of the tube
            target = (self.rod.copy()[:,0] + 1j * self.rod.copy()[:,1]) * e ** (-1j * np.pi/2) + 70
            self.target = np.stack((np.real(target), np.imag(target)), axis=1)

            # The tube with the rod and the particles on one side and the target on the other side
            self.rod[:,0] = self.rod[:,0] - 40
            self.particles[:,0] = self.particles[:,0] - 92

        elif self.obst_conf == "test_fr":

            # Add a wall parallel and close to the rod
            self.obstacles = self.rod.copy()
            self.obstacles[:,0] = self.obstacles[:,0] - 10

            # Point a few particles against the rod at an angle of 45°
            self.particles = np.array([[10, -20, 3*np.pi/4],
                                       [10, -30, 3*np.pi/4],
                                       [10, -40, 3*np.pi/4],
                                       [10, -10, 3*np.pi/4],
                                       [10, 0, 3*np.pi/4],
                                       [10, 10, 3*np.pi/4],
                                       [10, 20, 3*np.pi/4],
                                       [10, 30, 3*np.pi/4]])

        # Also give the number of obstacles
        self.n_obst = self.obstacles.shape[0]
        return


    def make_tube(self):
        '''
        Little helper to construct a tube
        '''

        # Get walls in the complex plane to build stuff
        little_wall = self.rod.copy()[30:,0] + 1j * self.rod.copy()[30:,1]
        little_wall_hor = little_wall * e ** (-1j * np.pi/2)

        # Make the tube
        inside = np.concatenate(
            (little_wall_hor - 25 + 1j * 20,
                little_wall_hor - 25 - 1j * 20), axis=0)
        left_up_face = np.concatenate(
            (little_wall - 25 + 1j * 20,
                little_wall - 25 + 1j * 70), axis=0)
        right_down_face = left_up_face * e ** (1j * np.pi)
        right_up_face = left_up_face + 50
        left_down_face = right_up_face * e ** (1j * np.pi)
        obstacles = np.concatenate((inside, left_up_face, left_down_face,
                                    right_up_face, right_down_face), axis=0)
        tube = np.stack((np.real(obstacles), np.imag(obstacles)), axis=1)

        return tube


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
        x_obst = self.obstacles[:,0]
        y_obst = self.obstacles[:,1]

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
        # CR_noise == 'no' means no noise at all, also not in the real simulation steps
        if self.CR_noise == 'no':
            noise_flag = 0
        else:
            noise_flag = 1

        self.particles, self.rod, self.part_rod_forces, \
        self.old_ther_noise, self.old_vel_noise, self.old_tor_noise = evolve.evolve_md_rod(fr_rod, inert_rod,
                                    X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                    x_rod, y_rod, x_obst, y_obst, self.dist_rod, actions,
                                    self.Rm, self.Rr, self.dt,
                                    self.torque, self.vel_act, self.vel_tor, self.vel_noise_fact, self.rot_noise_fact,
                                    self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                    noise_flag, int(self.use_obst), self.N, self.n_rod, self.n_obst, self.int_steps)

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
        x_obst = self.obstacles[:,0]
        y_obst = self.obstacles[:,1]
        if (self.mode == 4):
            assert rot_dir in [-1,1]
            assert old_rot_dir in [-1,1]

        # Now the observables are determined, if rew_mode=='torque' the rewards determined here are used, too
        obs, rew_torque, self.touch, self.rod_dist = evolve.get_o_r_rod(p[:,0], p[:,1], p[:,2],
                                          r[:,0], r[:,1], olr[:,0], olr[:,1], tar[:,0], tar[:,1],
                                          x_obst, y_obst,
                                          self.mode, rot_dir, old_rot_dir,
                                          self.flag_side, self.flag_LOS,
                                          self.part_size, self.part_size_rod,
                                          self.ext_rod, self.cen_rod,
                                          obs_type,
                                          self.cones, self.cone_angle, self.close_pen, self.prox_rew, self.flag_fix_or,
                                          int(self.obst_vision),
                                          self.n_obs, self.N, self.n_rod, self.n_obst)

        if self.rew_mode == 'torque':

            # Rewards based on position along and orientation with respect to the rod
            rewards = rew_torque

        elif self.rew_mode == 'team':

            # Determines rewards in team way (close? rotated?) So far only for rot.
            # the team_rew_mode specifies if touching or closeness are decisive.
            # In the case of approx_diff, an reward estimation during passive actions
            # is subtracted in learning_rod for approximating difference rewards.
            rewards = self.get_team_rewards(team_rew_mode=self.team_rew_mode)

        elif self.rew_mode == 'CR':

            # Determines the reward according to what would have happened if particle i would not have been there.
            # (Wonderful Life Utility, CR) It also uses a scaling that makes the experiment and the simulations
            # more compatible
            if self.parallelize_cr:
                rewards = self.get_CR_parallelized()
            else:
                rewards = self.get_CR()

        self.rewards = rewards

        return obs, rewards


    def get_team_rewards(self, team_rew_mode='touch'): # the team_rew_mode specifies if touching or closeness are decisive
        '''
        This simply rewards every particle that is present within a certain
        area around the rod if the rod has moved or rotated, etc.
        '''

        performance = self.det_performance(self.rod)

        # r = self.rod
        # olr = self.old_rod
        # rod_theta = np.angle(complex(r[-1,0] - r[0,0], r[-1,1] - r[0,1]))
        # d_theta_uncorr = rod_theta - np.angle(complex(olr[-1,0] - olr[0,0], olr[-1,1] - olr[0,1])) # Still can have jumps
        # d_theta = d_theta_uncorr - np.floor(d_theta_uncorr/(2 * np.pi) + 0.5) * 2 * np.pi # Now the jumps are corrected
        # performance = abs(d_theta)

        if self.mode == 3: # Rotation
            ref_prefactor = self.r_rew_fact

        if team_rew_mode == 'close':
            r = self.rod
            p = self.particles

            # Determining the distances to the rod (for every mode)
            p_comp = np.array(p[:,0] + 1j * p[:,1], ndmin=2)
            r_comp = np.array(r[:,0] + 1j * r[:,1], ndmin=2)

            dist = np.transpose(abs(p_comp - np.transpose(r_comp))) # Particles are in rows with their distances to the rod in columns
            min_dist = np.transpose(np.amin(dist, axis=1))
            close_enough = min_dist <= self.rew_cutoff # Only the particles within the cutoff distance to the rod get rewarded

            rewards = close_enough * performance * ref_prefactor # The direction of rotation does not matter.

        elif team_rew_mode == 'touch':

            rewards = self.touch * performance * ref_prefactor # The direction of rotation does not matter.

        elif team_rew_mode == 'team':

            rewards = np.full_like(self.touch, performance * ref_prefactor, dtype=np.double) # Really all particles get rewarded

        return rewards


    def get_CR(self):
        '''
        Determines the reward for particle i according to how the performance would have
        changed if particle i would not have been present (hypPerformance).
        This is as general as possible, while it is very simple,
        since all forces are considered automaticaly.
        Yes, this is computationally very expensive.
        '''

        # Mode 7 needs an exception here: If the rod reaches it's target
        # (within certain limits), all particles get a
        # high reward and the episode is stopped.
        if self.mode == 7:

            self.check_task_achieved()

            if self.task_achieved:

                # Particles all get a high reward (10)
                rewards = np.full(self.old_part.shape[0], self.final_rew)
                return rewards


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
        # for the case without noise (CR_noise = 'off') this is done from a noislessRod determined
        # from the last simulation step without noise
        if self.CR_noise == 'on' or self.CR_noise == 'mixed' or self.CR_noise == 'ideal' or self.CR_noise == 'no':

            perf_rod = self.rod

        elif self.CR_noise == 'off':
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
            x_obst = self.obstacles[:,0]
            y_obst = self.obstacles[:,1]

            # these are necessary due to the possibility to reproduce the last step with the old noise and reproduction == True
            reproduction = False
            old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
            old_vel_noise = np.zeros((self.N, self.int_steps))
            old_tor_noise = np.zeros((self.N, self.int_steps))

            _, perf_rod, _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                                X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                                x_rod, y_rod, x_obst, y_obst, self.dist_rod, action,
                                                self.Rm, self.Rr, self.dt,
                                                self.torque, self.vel_act, self.vel_tor, self.vel_noise_fact, self.rot_noise_fact,
                                                self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                                noise_flag, int(self.use_obst), N, self.n_rod, self.n_obst, self.int_steps)

        performance = self.det_performance(perf_rod)

        # The hyp_perf are the hypothetical performances that would have been achieved in the absence of particle i
        hyp_perf, hyp_rod_ang, hyp_rod, hyp_parts = self.det_hyp_perf(performance)

        # The contribution of a particle is the difference between the actual performance
        # and the hypothetical performance if it would not have been there.
        contrib = performance - hyp_perf

        # Really with performance here? Yes, because otherwise opposing particles get both rewarded even though nothing happens.
        # Wolpert and Tumer (2001) do not multiply the performance here.
        rewards = self.CR_prefact * contrib

        # To encourage particles to interact with the rods,
        # all particles touching the rod get a small reward
        # (should be much smaller than the rewards generated
        # by pushing the rod to the target, 0.1 should be fine)
        if self.mode == 7:
            rewards[self.touch == 1] = rewards[self.touch == 1] + self.CR_touch_rew

        # If required do not use rewards but a cost. This is achieved by subtracting
        # an estimate of the optimal reward a particle has (from looking at
        # previous data: the average maximum reward is about 1) from the actual reward,
        # such that in the optimal case a reward of about 0 is reached
        if self.cost_iso_rew:
            rewards = rewards - 1

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


    def get_CR(self):
        '''
        Determines the reward for particle i according to how the performance would have
        changed if particle i would not have been present (hypPerformance).
        This is as general as possible, while it is very simple,
        since all forces are considered automaticaly.
        Yes, this is computationally very expensive.
        This version of it enforces the mixed noise mode
        '''

        # Because that's the only working noise mode for the experiment
        self.CR_noise = 'mixed'

        # Mode 7 needs an exception here: If the rod reaches it's target
        # (within certain limits), all particles get a
        # high reward and the episode is stopped.
        if self.mode == 7:

            self.check_task_achieved()

            if self.task_achieved:

                # Particles all get a high reward (10)
                rewards = np.full(self.old_part.shape[0], self.final_rew)
                return rewards

        # In the initialization, determining this type of reward is not possible
        if sum(self.old_actions) <= 0:
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
        N = self.old_part.shape[0]
        x_rod = self.old_rod[:,0]
        y_rod = self.old_rod[:,1]
        fr_rod = self.fr_rod
        inert_rod = self.inert_rod
        x_obst = self.obstacles[:,0]
        y_obst = self.obstacles[:,1]

        # these are necessary due to the possibility to reproduce the last step with the old noise and reproduction == True
        reproduction = False
        old_ther_noise = np.zeros((N, 3 * self.int_steps))
        old_vel_noise = np.zeros((N, self.int_steps))
        old_tor_noise = np.zeros((N, self.int_steps))

        _, virtual_rod, _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                            X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                            x_rod, y_rod, x_obst, y_obst, self.dist_rod, action,
                                            self.Rm, self.Rr, self.dt,
                                            self.torque, self.vel_act, self.vel_tor, self.vel_noise_fact, self.rot_noise_fact,
                                            self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                            noise_flag, int(self.use_obst), N, self.n_rod, self.n_obst, self.int_steps)

        virtual_performance = self.det_performance(virtual_rod)
        experiment_performance = self.det_performance(experiment_rod)

        # The hyp_perf are the hypothetical performances that would have been achieved in the absence of particle i
        # The experiment performance is given here, because that is the baseline
        hyp_perf, hyp_rod_ang, hyp_rod, hyp_parts = self.det_hyp_perf(virtual_performance)

        # The contribution of a particle is the difference between the actual performance
        # and the hypothetical performance if it would not have been there,
        # scaled such that if the effect of this particle tends to 0, the performance
        # matches the experimental performance
        if virtual_performance == 0 or np.any(abs(hyp_perf/virtual_performance) >= 10):
            contrib = experiment_performance - hyp_perf
        else:
            contrib = experiment_performance - hyp_perf * experiment_performance/virtual_performance

        # Wolpert and Tumer (2001) do not multiply with the performance here.
        rewards = self.CR_prefact * contrib

        # To encourage particles to interact with the rods,
        # all particles touching the rod get a small reward
        # (should be much smaller than the rewards generated
        # by pushing the rod to the target, 0.1 should be fine)
        if self.mode == 7:
            rewards[self.touch == 1] = rewards[self.touch == 1] + self.CR_touch_rew

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


    def get_CR_parallelized(self):
        '''
        Determines the reward for particle i according to how the performance would have
        changed if particle i would not have been present (hypPerformance).
        This is as general as possible, while it is very simple,
        since all forces are considered automaticaly.
        Yes, this is computationally very expensive.
        This version uses a pool (from multiprocessing) for that
        This version of it enforces the mixed noise mode
        '''

        # Because that's the only working noise mode for the experiment
        self.CR_noise = 'mixed'

        # Mode 7 needs an exception here: If the rod reaches it's target
        # (within certain limits), all particles get a
        # high reward and the episode is stopped.
        if self.mode == 7:

            self.check_task_achieved()

            if self.task_achieved:

                # Particles all get a high reward (10)
                rewards = np.full(self.old_part.shape[0], self.final_rew)
                return rewards

        # In the initialization, determining this type of reward is not possible
        if sum(self.old_actions) <= 0:
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
        x_obst = self.obstacles[:,0]
        y_obst = self.obstacles[:,1]

        # these are necessary due to the possibility to reproduce the last step with the old noise and reproduction == True
        reproduction = False
        old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
        old_vel_noise = np.zeros((self.N, self.int_steps))
        old_tor_noise = np.zeros((self.N, self.int_steps))

        _, virtual_rod, _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                            X, Y, T, old_ther_noise, old_vel_noise, old_tor_noise,
                                            x_rod, y_rod, x_obst, y_obst, self.dist_rod, action,
                                            self.Rm, self.Rr, self.dt,
                                            self.torque, self.vel_act, self.vel_tor, self.vel_noise_fact, self.rot_noise_fact,
                                            self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                            noise_flag, int(self.use_obst), N, self.n_rod, self.n_obst, self.int_steps)

        virtual_performance = self.det_performance(virtual_rod)
        experiment_performance = self.det_performance(experiment_rod)

        # Start a pool to determine the hypothetical performance for many
        # particles in parallel
        self.virtual_performance = virtual_performance
        with Pool(processes=self.n_processes) as pool:
            par_results = tuple(pool.map(self.det_hyp_perf_one_part, range(self.particles.shape[0])))

        # Concatenate the results of the hypothetical performance computation
        hyp_perf = np.asarray([p[0] for p in par_results])
        hyp_rod_ang = np.asarray([p[1] for p in par_results])
        hyp_rod = np.asarray([p[2] for p in par_results]).transpose((1, 2, 0))
        hyp_parts = np.asarray([p[3] for p in par_results]).transpose((1, 2, 0))

        # For debugging:
        # Get the hypothetical performance without parallelization
        # hyp_perf_normal, hyp_rod_ang_normal, hyp_rod_normal, hyp_parts_normal = self.det_hyp_perf(virtual_performance)

        # Determine hyp perf with loop instead of parallelization
        # hyp_perf_loop = np.full_like(hyp_perf_normal, 0)
        # hyp_rod_ang_loop = np.full_like(hyp_rod_ang_normal, 0)
        # hyp_rod_loop = np.full_like(hyp_rod_normal, 0)
        # hyp_parts_loop = np.full_like(hyp_parts_normal, 0)
        # for i in range(self.particles.shape[0]):
        #     hyp_perf_loop[i], hyp_rod_ang_loop[i], _, _ = self.det_hyp_perf_one_part(i)

        # Give a warning when the hypothetical performances of the parallelized version
        # and the normal version don't match
        # if not all(hyp_perf == hyp_perf_normal):
        #     print("Performances in parallelized version and non-parallelized version of det_hyp_perf don't match")

        # The contribution of a particle is the difference between the actual performance
        # and the hypothetical performance if it would not have been there,
        # scaled such that if the effect of this particle tends to 0, the performance
        # matches the experimental performance
        if virtual_performance == 0 or np.any(abs(hyp_perf/virtual_performance) >= 10):
            contrib = experiment_performance - hyp_perf
        else:
            contrib = experiment_performance - hyp_perf * experiment_performance/virtual_performance

        # Wolpert and Tumer (2001) do not multiply with the performance here.
        rewards = self.CR_prefact * contrib

        # To encourage particles to interact with the rods,
        # all particles touching the rod get a small reward
        # (should be much smaller than the rewards generated
        # by pushing the rod to the target, 0.1 should be fine)
        if self.mode == 7:
            rewards[self.touch == 1] = rewards[self.touch == 1] + self.CR_touch_rew

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
            # A "value" (in the form if a potential) is calculated
            # from relative position of rod and target.
            # The "values" of a certain rod position exponentially
            # decreases from the target. The performance is then
            # determined by the change in the "value" of the rod.

            # Complex representations of everything (target, rod and old rod)
            tar_c = t[:,0] + 1j *  t[:,1]
            rod_c = r[:,0] + 1j *  r[:,1]
            olr_c = olr[:,0] + 1j *  olr[:,1]

            # Determining the distances between rod and target particles.
            # This can be done PARallel or ANTIparallel and the version
            # with the smaller distances is chosen (further down)
            dists_new_par = abs(tar_c - rod_c)
            dists_new_anti = abs(tar_c - np.flip(rod_c))
            dists_old_par = abs(tar_c - olr_c)
            dists_old_anti = abs(tar_c - np.flip(olr_c))

            # The value is linearly decreasing from the target.
            value_new = max(
                sum(-dists_new_par),
                sum(-dists_new_anti))
            value_old = max(
                sum(-dists_old_par),
                sum(-dists_old_anti))

            # The values get normalized by a constant,
            # such that the performance gives proper rewards
            norm_constant = self.n_rod

            # determining the performance from the change in the value
            # (without subtracting the old value, since this would give a reward of 0 when the particles have achieved their goal)
            performance = (value_new - value_old) / norm_constant

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

        if self.CR_noise == 'on':
            noise_flag = 1
        elif self.CR_noise == 'off' or self.CR_noise == 'mixed' or self.CR_noise == 'ideal' or self.CR_noise == 'no':
            noise_flag = 0


        # in the CR_noise 'ideal', the same noise as in the last step is used to determine the hypPerfs
        if self.CR_noise == 'ideal':
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

        # Iterate over every particle, leave out that particle (CR_mode = 'non_ex')
        # or make it passive (CR_mode = 'passive') and simulate one step.
        for i in range(self.particles.shape[0]):

            if (self.CR_rew_mode == 'close' and (distances[i] <= self.rew_cutoff)) or (self.CR_rew_mode == 'touch' and touch[i]):

                if self.CR_mode == 'non_ex':
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

                elif self.CR_mode == 'passive':
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

                # Get the obstacle coordinates
                x_obst = self.obstacles[:,0]
                y_obst = self.obstacles[:,1]

                hyp_parts[mask,:,i], hyp_rod[:,:,i], _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                                X, Y, T, old_th_n, old_v_n, old_tor_n,
                                                x_rod, y_rod, x_obst, y_obst, self.dist_rod, action,
                                                self.Rm, self.Rr, self.dt,
                                                self.torque, self.vel_act, self.vel_tor, self.vel_noise_fact, self.rot_noise_fact,
                                                self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                                noise_flag, int(self.use_obst), N, self.n_rod, self.n_obst, self.int_steps)

                if self.CR_mode == "non_ex":
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


    def det_hyp_perf_one_part(self, i):
        '''
        This determines for one particle, how the performance would have been in
        the absence of this particle. It needs the old particle and rod positions
        as well as the old actions.
        For every particle, it evolves the environment one step without this particle.
        '''

        performance = self.virtual_performance

        # Set the noise for determining the hypothetical performances:
        # 'on':    noise in determining performance and hypPerf
        # 'off':   no noise in determining perf and hypPerf
        # 'mixed': noise in determining perf and no noise in determining hypPerf
        # 'ideal': exactly the same noise in both perf and hypPerf calculations

        if self.CR_noise == 'on':
            noise_flag = 1
        elif self.CR_noise == 'off' or self.CR_noise == 'mixed' or self.CR_noise == 'ideal' or self.CR_noise == 'no':
            noise_flag = 0

        # in the CR_noise 'ideal', the same noise as in the last step is used to determine the hypPerfs
        if self.CR_noise == 'ideal':
            reproduction = True
            old_ther_noise = self.old_ther_noise
            old_vel_noise = self.old_vel_noise
            old_tor_noise = self.old_tor_noise
        else:
            reproduction = False
            old_ther_noise = np.zeros((self.N, 3 * self.int_steps))
            old_vel_noise = np.zeros((self.N, self.int_steps))
            old_tor_noise = np.zeros((self.N, self.int_steps))


        # Define defaults for the output
        hyp_perf = 0
        hyp_rod_ang = 0 # this is just the angle, 0 as default is bad, I know
        hyp_rod = np.zeros((self.rod.shape[0], self.rod.shape[1]))
        hyp_parts = np.zeros((self.old_part.shape[0], self.old_part.shape[1]))

        # Define the starting configuretion
        x_rod = self.old_rod[:,0]
        y_rod = self.old_rod[:,1]
        fr_rod = self.fr_rod
        inert_rod = self.inert_rod

        # Get the obstacle coordinates
        x_obst = self.obstacles[:,0]
        y_obst = self.obstacles[:,1]

        distance = self.rod_dist[i]
        touch = self.touch[i]

        # Leave out that particle (CR_mode = 'non_ex')
        # or make it passive (CR_mode = 'passive') and simulate one step.

        if (self.CR_rew_mode == 'close' and (distance <= self.rew_cutoff)) or (self.CR_rew_mode == 'touch' and touch):

            if self.CR_mode == 'non_ex':
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

            elif self.CR_mode == 'passive':
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

            hyp_parts[mask,:], hyp_rod[:,:], _, _, _, _ = evolve.evolve_md_rod(fr_rod, inert_rod,
                                            X, Y, T, old_th_n, old_v_n, old_tor_n,
                                            x_rod, y_rod, x_obst, y_obst, self.dist_rod, action,
                                            self.Rm, self.Rr, self.dt,
                                            self.torque, self.vel_act, self.vel_tor, self.vel_noise_fact, self.rot_noise_fact,
                                            self.ext_rod, self.cen_rod, self.mu_K, reproduction,
                                            noise_flag, int(self.use_obst), N, self.n_rod, self.n_obst, self.int_steps)

            # If the particle is made to not exist, we have to add it to the
            # hypothetical particles anyway
            if self.CR_mode == "non_ex":
                hyp_parts[i,:] = self.old_part[i,:]

            # Now the hypPerformance in the absence of particle i is determined
            hyp_perf = self.det_performance(hyp_rod[:,:])

        else:
            # If the particle is too far away to have an effect (distance > rew_cutoff),
            # the performance without this particle should be the same as with it.
            hyp_perf = performance
            # the hypothetical particles and rod must still be written
            hyp_rod[:,:] = self.rod
            hyp_parts[:,:] = self.old_part

            # for debugging, the hypothetical rod angles are also returned
            hyp_rod_ang = np.angle(complex(hyp_rod[-1,0] - hyp_rod[0,0], hyp_rod[-1,1] - hyp_rod[0,1]))

        return hyp_perf, hyp_rod_ang, hyp_rod, hyp_parts


    def check_task_achieved(self):

        assert self.termination_mode == "sum" or self.termination_mode == "ind", "No termination mode selected"

        # Calculate the sum of the rod-target distances
        r = self.rod
        t = self.target
        tar_c = t[:,0] + 1j *  t[:,1]
        rod_c = r[:,0] + 1j *  r[:,1]

        # Determine, which way round to consider the target (since there are two possible "orientations")
        if sum(abs(tar_c - np.flip(rod_c))) < sum(abs(tar_c - rod_c)):
            rod_c = np.flip(rod_c)

        if self.termination_mode == "sum":

            # The sum of rod to target diatances needs to be small
            cumm_dists = sum(abs(tar_c - rod_c))
            if cumm_dists < self.n_rod * self.achieved_dist:
                self.task_achieved = True

        elif self.termination_mode == "ind":

            # All individual rod particles need to be close to the target
            if np.all(abs(tar_c - rod_c) < self.achieved_dist):
                self.task_achieved = True

#
# ------------------------------------
# End of class MD
# ------------------------------------
