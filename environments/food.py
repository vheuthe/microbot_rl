
import numpy as np
from fortran import evolve_food

class FoodEnvironment():
    """Environment to simulate active swimmers in different food scenarios"""

    def __init__(self, food_mode, dt, action_time, Dt, Dr, vel_act, sig_vel_act, vel_tor, sig_vel_tor, torque,
                 input_dim, food_rew, touch_penalty, max_nn_rew, cones, cone_angle, visual_particle_size, obs_type, obs_noise,
                 food_dist, food_amount, food_width, food_delay,
                 **parameters):

        # Time resolution
        self.dt = dt
        self.steps = int(action_time / dt)
        # Brownian dynamics
        self.Dt = Dt
        self.Dr = Dr
        self.Rm = np.sqrt(2*self.Dt/self.dt)
        self.Rr = np.sqrt(2*self.Dr/self.dt)
        # Active Properties
        self.vel_act = vel_act
        self.sig_vel_act = sig_vel_act
        self.vel_tor = vel_tor
        self.sig_vel_tor = sig_vel_tor
        self.torque = 1.0 / 350.0 * torque # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)
        # Food configuration
        self.food_dist = food_dist
        self.food_amount = food_amount
        self.food_width = food_width
        self.food_delay = food_delay
        # Obervables and Rewards
        self.n_obs = input_dim
        self.cones = cones
        self.cone_angle = cone_angle / 180 * np.pi
        self.visual_particle_size = visual_particle_size
        self.obs_type = {'1overR': 1, '1overR2': 2}[obs_type]
        self.obs_noise = obs_noise
        self.food_rew = food_rew
        self.touch_penalty = touch_penalty
        self.max_nn_rew = max_nn_rew
        # everything that was not catched by the arguments
        self.parameters = parameters

        # Current State
        self.foodrng = None
        # n*(x,y,θ)
        self.particles = []
        # n*(x,y,amount,width,delaycounter, ...)
        self.food = []
        self.food_counter = 0

        # register functions for choosen food scenario
        self.reset_food = {
            'random': self.reset_food_random,
            'alternating': self.reset_food_alternating,
            'none': self.reset_food_none,
            '2sources': self.reset_food_2sources,
            'randombox': self.reset_food_randombox,
        }[food_mode]
        self.update_food = {
            'random': self.update_food_random,
            'alternating': self.update_food_alternating,
            'none': self.update_food_none,
            '2sources': self.update_food_2sources,
            'randombox': self.update_food_randombox,
        }[food_mode]


    def reset(self, n, seed=None):
        """Initializes the environment with n particles and new food"""

        # set up rng, this is important for reproducibility of evaluation runs
        self.foodrng = np.random.default_rng(seed)
        self.obsrng = np.random.default_rng(seed)

        # generate grid position in a random order
        a = np.int(np.sqrt(n) / 2) + 1
        x, y = np.meshgrid(range(-a, a+1), range(-a, a+1))
        pos = np.array(list(zip(x.flat, y.flat)))
        self.foodrng.shuffle(pos)

        # distribute particles with random θ on the first n positions
        self.particles = np.append(
            20 * pos[0:n, :],
            2 * np.pi * self.foodrng.random((n, 1)),
            axis=1
        )

        # (re)initialize food sources
        self.reset_food()

        # compute observables for initial state s_0
        observables, _reward, _eaten = self.get_state()

        return observables


    def evolve(self, actions):
        """Simulates the environment dynamics p(s', r | s, a)

        As the Environment keeps track of the current state, only the choosen actions have to be
        passed to the function.
        """

        # Evolve multiple steps of brownian dynamics for one action
        self.particles = evolve_food.evolve_md(
            self.particles[:,0], self.particles[:,1], self.particles[:,2], actions,
            self.Rm, self.Rr, self.dt, self.steps, self.torque,
            self.vel_act, self.sig_vel_act, self.vel_tor, self.sig_vel_tor
        )

        # Compute Observables and reward r for new state s'
        observables, reward, eaten = self.get_state()

    	# If food got depleeted, it might need to be relocated
        self.update_food(eaten)

        return observables, reward

    def get_state(self):
        observables, reward, eaten = evolve_food.get_o_r_food_task(
            self.particles[:,0], self.particles[:,1], self.particles[:,2],
            self.obs_type, self.cone_angle, 0, self.food_rew, self.touch_penalty,
            self.food[:,0], self.food[:,1], self.food[:,2], self.food[:,3] * (self.food[:,2] > 0),
            self.max_nn_rew, self.visual_particle_size, 4 * self.cones,
        )

        # Add noise (clip those observables that are positive only)
        observables += self.obsrng.normal(0, self.obs_noise, observables.shape)
        observables[:,[0,3,6,9,12,15,16,17,18,19]] = observables[:,[0,3,6,9,12,15,16,17,18,19]].clip(0, None)

        # Reduce Information
        if self.n_obs == 5:
            observables = observables[:,15:]
        elif self.n_obs == 10:
            observables = observables[:,[0,3,6,9,12,15,16,17,18,19]]

        return observables, reward, eaten


    # - - - different food scenarios - - -

    # alternating between +/- food_dist/2
    def reset_food_alternating(self):
        self.food = np.array([[self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay]])
        self.food_counter = 1

    def update_food_alternating(self, eaten):
        self.food[0,2] -= eaten[0]
        if self.food[0,2] < 1:
            self.food[0,4] -= 1
            if self.food[0,4] < 1:
                self.food[0,0] = - np.sign(self.food[0,0]) * self.food_dist / 2
                self.food[0,2] = self.food_amount
                self.food[0,4] = self.food_delay
                self.food_counter += 1

    # random new position at food_dist (+/-food_dist/3) from old position
    def reset_food_random(self):
        self.food = np.array([[self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay]])
        self.food_counter = 1

    def update_food_random(self, eaten):
        self.food[0,2] -= eaten[0]
        if self.food[0,2] < 1:
            self.food[0,4] -= 1
            self.food[0,3] = 0
            if self.food[0,4] < 1:
                phi = self.foodrng.random()*np.pi*2
                displ = self.foodrng.normal(self.food_dist, self.food_dist/3)
                self.food[0,0] += displ * np.cos(phi)
                self.food[0,1] += displ * np.sin(phi)
                self.food[0,2] = self.food_amount
                self.food[0,3] = self.food_width
                self.food[0,4] = self.food_delay
                self.food_counter += 1

    # no food, for steady state analysis (meant to be used without training)
    def reset_food_none(self):
        self.food = np.zeros((1,4))

    def update_food_none(self, _eaten):
        pass

    # no food, for steady state analysis (meant to be used without training)
    def reset_food_2sources(self):
        self.food = np.array([
            [self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay],
            [self.food_width/2-self.food_dist, 0, self.food_amount, self.food_width, self.food_delay]
        ])
        self.food_counter = 2

    def update_food_2sources(self, eaten):
        for i in range(eaten.shape[0]):
            self.food[i,2] -= eaten[i]
            if self.food[i,2] < 1:
                self.food[i,3] = 0
                self.food[i,4] -= 1
                if self.food[i,4] < 1:
                    j = (i + 1) % 2
                    # angle to 2nd food source +- 60°
                    phi = np.arctan2(
                        self.food[j,1] - self.food[i,1],
                        self.food[j,0] - self.food[i,0]
                    ) + self.foodrng.choice([np.pi/3, -np.pi/3])
                    # new food forms a triangle with 2nd and depletet food source
                    self.food[i,0] += self.food_dist * np.cos(phi)
                    self.food[i,1] += self.food_dist * np.sin(phi)
                    self.food[i,2] = self.food_amount
                    self.food[i,3] = self.food_width
                    self.food[i,4] = self.food_delay
                    self.food_counter += 1

    # random within a box comparable to the experiment
    def reset_food_randombox(self):
        self.food = np.array([[self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay]])
        self.food_counter = 1

    def update_food_randombox(self, eaten):
        self.food[0,2] -= eaten[0]
        if self.food[0,2] < 1:
            self.food[0,4] -= 1
            self.food[0,3] = 0
            if self.food[0,4] < 1:
                self.food[0,0] = self.foodrng.uniform(-150, 150)
                self.food[0,1] = self.foodrng.uniform(-100, 100)
                self.food[0,2] = self.food_amount
                self.food[0,3] = self.food_width
                self.food[0,4] = self.food_delay
                self.food_counter += 1