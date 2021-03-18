
import numpy as np
import evolve_fortran_discreteFood as evolve_food


class FoodEnvironment():
    """Environment to simulate active swimmers in different food scenarios"""

    def __init__(self, food_mode, dt, action_time, vel_act, sig_vel_act, vel_tor, sig_vel_tor, torque,
                 food_rew, touch_penalty, cones, cone_angle, food_dist, food_amount, food_width, food_delay, 
                 **parameters):

        # Time resolution
        self.dt = dt
        self.steps = int(action_time / dt)
        # Brownian dynamics
        self.Dt = 0.014
        self.Dr = 1.0 / 350.0
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
        self.cones = cones
        self.cone_angle = cone_angle / 180 * np.pi
        self.food_rew = food_rew
        self.touch_penalty = touch_penalty
        # everything that was not catched by the arguments
        self.parameters = parameters

        # Current State
        # n*(x,y,θ)
        self.particles = []
        # n*(x,y,amount,width,delaycounter, ...)
        self.food = []

        # register functions for choosen food scenario
        self.reset_food = {
            'random': self.reset_food_random,
            'alternating': self.reset_food_alternating,
            'none': self.reset_food_none,
        }[food_mode]
        self.update_food = {
            'random': self.update_food_random,
            'alternating': self.update_food_alternating,
            'none': self.update_food_none,
        }[food_mode]


    def reset(self, n):
        """Initializes the environment with n particles and new food"""

        # generate grid position in a random order
        a = np.int(np.sqrt(n) / 2) + 1
        x, y = np.meshgrid(range(-a, a+1), range(-a, a+1))
        pos = np.array(list(zip(x.flat, y.flat)))
        np.random.shuffle(pos)

        # distribute particles with random θ on the first n positions
        self.particles = np.append(
            20 * pos[0:n, :],
            2 * np.pi * np.random.rand(n, 1),
            axis=1
        )

        # (re)initialize food sources
        self.reset_food()

        # compute observables for initial state s_0
        observables, _reward, _eaten = evolve_food.get_o_r_food_task(
            self.particles[:,0], self.particles[:,1], self.particles[:,2], 
            1, self.cone_angle, 0, self.food_rew, self.touch_penalty, 
            self.food[0,0], self.food[0,1], self.food[0,2], self.food[0,3] * (self.food[0,2] > 0),
            999, 6.2, 4 * self.cones,
        )

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
        observables, reward, eaten = evolve_food.get_o_r_food_task(
            self.particles[:,0], self.particles[:,1], self.particles[:,2], 
            1, self.cone_angle, 0, self.food_rew, self.touch_penalty, 
            self.food[0,0], self.food[0,1], self.food[0,2], self.food[0,3] * (self.food[0,2] > 0),
            999, 6.2, 4 * self.cones,
        )

    	# If food got depleeted, it might need to be relocated
        self.update_food(eaten)

        return observables, reward


    # - - - different food scenarios - - -

    # alternating between +/- food_dist/2
    def reset_food_alternating(self):
        self.food = np.array([[self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay]])

    def update_food_alternating(self, eaten):
        self.food[0,2] -= eaten
        if self.food[0,2] < 1:
            self.food[0,4] -= 1
            if self.food[0,4] < 1:
                self.food[0,0] = - np.sign(self.food[0,0]) * self.food_dist / 2
                self.food[0,2] = self.food_amount
                self.food[0,4] = self.food_delay

    # random new position at food_dist (+/-food_dist/3) from old position
    def reset_food_random(self):
        self.food = np.array([[self.food_width/2, 0, self.food_amount, self.food_width, self.food_delay]])

    def update_food_random(self, eaten):
        self.food[0,2] -= eaten
        if self.food[0,2] < 1:
            self.food[0,4] -= 1
            if self.food[0,4] < 1:
                phi = np.random.rand()*np.pi*2
                displ = np.random.normal(self.food_dist, self.food_dist/3)
                self.food[0,0] += displ * np.cos(phi)
                self.food[0,1] += displ * np.sin(phi)
                self.food[0,2] = self.food_amount
                self.food[0,4] = self.food_delay

    # no food, for steady state analysis (meant to be used without training)
    def reset_food_none(self):
        self.food = np.zeros((1,4))

    def update_food_none(self, _eaten):
        pass

