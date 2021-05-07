
import numpy as np
from fortran import evolve_food



class AbstractFood():

    def __init__(self, food_mode, food_dist, food_amount, food_width, food_delay,
                 input_dim, food_rew, touch_penalty, tp_type, max_nn_rew, cones,
                 rew_cones, vision_angle, particle_size, visual_particle_size, obs_type,
                 **parameters):

        # Food configuration
        self.food_dist = food_dist
        self.food_amount = food_amount
        self.food_width = food_width
        self.food_delay = food_delay
        # Obervables and Rewards
        self.n_obs = input_dim
        self.cones = cones
        self.vision_angle = vision_angle / 180 * np.pi
        self.particle_size = particle_size
        self.visual_particle_size = visual_particle_size
        self.obs_type = {'1overR': 1, '1overR2': 2}[obs_type]
        self.food_rew = food_rew
        self.touch_penalty = touch_penalty
        self.tp_type = {'all': 1, 'closest': 2, '1overR3': 3}[tp_type]
        self.max_nn_rew = max_nn_rew
        self.rew_cones = rew_cones

        # Current State
        self.foodrng = None
        # n*(x,y,amount,width,delaycounter, ...)
        self.food = []
        self.food_counter = 0

        # register functions for choosen food scenario
        self.reset_food = {
            'random': self.reset_food_one,
            'alternating': self.reset_food_one,
            'none': self.reset_food_none,
            '2sources': self.reset_food_two,
            'experiment': self.reset_food_one,
            '2inexp': self.reset_food_two,
        }[food_mode]
        self.update_food = {
            'random': self.update_food_random,
            'alternating': self.update_food_alternating,
            'none': self.update_food_none,
            '2sources': self.update_food_2sources,
            'experiment': self.update_food_experiment,
            '2inexp': self.update_food_2inexp,
        }[food_mode]

        # reset for convenience
        self.reset()

    def reset(self, seed=None):
        # set up rng, this is important for reproducibility of evaluation runs
        self.foodrng = np.random.default_rng(seed)
        # (re)initialize food sources depending on type
        self.reset_food()

    def update(self, particles):
        # Compute Observables and reward r for new state s'
        observables, reward, eaten = self.get_state(particles)
    	# If food got depleeted, it might need to be relocated
        self.update_food(eaten)

        return observables, reward

    def get_state(self, particles):

        # get_o_r_food_task(X, Y, Theta, XFood, YFood, RFood, &
        #     vision_angle, cones, dead_vision, obs_type, phys_size, vis_size, &
        #     food_rew, nn_rew_cones, max_nn_rew, tp_type, touch_penalty, &
        #     N, NFood, Obs, Rew, Eaten)
        observables, reward, eaten = evolve_food.get_o_r_food_task(
            particles[:,0], particles[:,1], particles[:,2],
            self.food[:,0], self.food[:,1], 0.5 * self.food[:,3] * (self.food[:,2] > 0),
            self.vision_angle, self.cones, 0, self.obs_type, self.particle_size, self.visual_particle_size,
            self.food_rew, self.rew_cones, self.max_nn_rew, self.tp_type, self.touch_penalty,
        )

        # Reduce Information
        if self.n_obs == 5:
            observables = observables[:,15:]
        elif self.n_obs == 10:
            observables = observables[:,[0,3,6,9,12,15,16,17,18,19]]

        return observables, reward, eaten


    # - - - different food scenarios - - -

    def reset_food_none(self):
        self.food = np.zeros((1,4))

    def reset_food_one(self):
        self.food = np.array([[self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay]])
        self.food_counter = 1

    def reset_food_two(self):
        self.food = np.array([
            [self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay],
            [self.food_width/2-self.food_dist, 0, self.food_amount, self.food_width, self.food_delay]
        ])
        self.food_counter = 2


    # no food, for steady state analysis (meant to be used without training)
    def update_food_none(self, _eaten):
        pass

    # alternating between +/- food_dist/2
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
    def update_food_experiment(self, eaten):
        self.food[0,2] -= eaten[0]
        if self.food[0,2] < 1:
            self.food[0,4] -= 1
            self.food[0,3] = 0
            if self.food[0,4] < 1:
                old_x = self.food[0,0]
                old_y = self.food[0,1]
                # redraw until distance is at least 'food_dist'
                while (self.food[0,0] - old_x)**2 + (self.food[0,1] - old_y)**2 < self.food_dist**2:
                    self.food[0,0] = self.foodrng.uniform(-100, 100)
                    self.food[0,1] = self.foodrng.uniform(-60, 60)
                self.food[0,2] = self.food_amount
                self.food[0,3] = self.food_width
                self.food[0,4] = self.food_delay
                self.food_counter += 1

    # 2 sources random within a box comparable to the experiment
    def update_food_2inexp(self, eaten):
        for i in range(eaten.shape[0]):
            self.food[i,2] -= eaten[i]
            if self.food[i,2] < 1:
                self.food[i,3] = 0
                self.food[i,4] -= 1
                if self.food[i,4] < 1:
                    # draw new position
                    self.food[i,0] = self.foodrng.uniform(-100, 100)
                    self.food[i,1] = self.foodrng.uniform(-60, 60)
                    # if necessary, redraw until distance is at least 'food_dist'
                    while (self.food[0,0] - self.food[1,0])**2 + (self.food[0,1] - self.food[1,1])**2 < self.food_dist**2:
                        self.food[i,0] = self.foodrng.uniform(-100, 100)
                        self.food[i,1] = self.foodrng.uniform(-60, 60)
                    # set the rest
                    self.food[i,2] = self.food_amount
                    self.food[i,3] = self.food_width
                    self.food[i,4] = self.food_delay
                    self.food_counter += 1




class FoodSimulation(AbstractFood):
    """Environment to simulate active swimmers in different food scenarios"""

    def __init__(self, dt, action_time, vel_act, vel_tor, vel_noise, torque, torque_noise, **parameters):

        # setup food
        super().__init__(**parameters)

        # Time resolution
        self.dt = dt
        self.steps = int(action_time / dt)

        # Active Properties
        self.vel_act = vel_act
        self.vel_tor = vel_tor
        self.vel_noise = vel_noise
        self.torque = 1.0 / 350.0 * torque # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)
        self.torque_noise = torque_noise

        # n*(x,y,θ)
        self.particles = []


    def reset(self, n, seed=None):
        """Initializes the environment with n particles and new food"""

        super().reset(seed)

        rng = np.random.default_rng(seed)

        # generate grid position in a random order
        a = np.int(np.sqrt(n) / 2) + 1
        x, y = np.meshgrid(range(-a, a+1), range(-a, a+1))
        pos = np.array(list(zip(x.flat, y.flat)))
        rng.shuffle(pos)

        # distribute particles with random θ on the first n positions
        self.particles = np.append(
            20 * pos[0:n, :],
            2 * np.pi * rng.random((n, 1)),
            axis=1
        )

        # compute observables for initial state s_0
        observables, _reward, _eaten = self.get_state(self.particles)

        return observables


    def evolve(self, actions):
        """Simulates the environment dynamics p(s', r | s, a)

        As the Environment keeps track of the current state, only the choosen actions have to be
        passed to the function.
        """

        # Evolve multiple steps of brownian dynamics for one action
        self.particles = evolve_food.evolve_md(
            self.particles, actions, self.steps, self.dt,
            self.torque, self.torque_noise, self.vel_act, self.vel_tor, self.vel_noise
        )

        return self.update(self.particles)
