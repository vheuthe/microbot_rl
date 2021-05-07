
import learning_food
from environments.food import FoodSimulation
import numpy as np
from scipy.io import savemat

env = FoodSimulation(**{**learning_food.default_parameters, 'food_rew': 0, 'touch_penalty': 0, 'food_mode': 'none'})
env.reset(1)

def get_reward(others):
    env.particles = np.append(np.zeros((1,3)), others, axis=0)
    _obs, rew, _eaten = env.get_state()
    return rew[0]

data = {}

# construct crystal:
# include particles "behind" 0|0 for touch penalty
xx, yy = np.meshgrid(range(-1, 10), range(-1, 10))
x = xx.reshape(-1,1) * np.cos(-np.pi/6)
y = yy.reshape(-1,1) + xx.reshape(-1,1) * np.sin(-np.pi/6)
# "remove" 0|0 (will get placed in front later)
origin = ((x == 0) & (y == 0))
x[origin] = 9999
y[origin] = 9999


for tp in [0, 1, 2]:

    data["TP" + str(tp)] = []
    env.touch_penalty = tp

    for d in np.arange(6.2, 24.8, 0.05):
        # single cone center:
        r1 = get_reward([[d, 0, 0]])
        # single particle cone boundary
        r2 = get_reward([[d * np.cos(np.pi/10), d * np.sin(np.pi/10), 0]])
        # crystal of perfectly spaced particles
        r3 = get_reward(np.hstack((d * x, d * y, np.zeros(x.shape))))
        # line of particles:
        dphi = 2*np.arcsin(3.1/d)
        angles = np.append(np.arange(-dphi,-np.pi,-dphi), np.arange(0, np.pi, dphi)).reshape(-1,1)
        r4 = get_reward(np.hstack((d * np.cos(angles), d * np.sin(angles), np.zeros(angles.shape))))

        data["TP" + str(tp)].append([d,r1,r2,r3,r4])

savemat('/tmp/test_food_rewards.mat', data)