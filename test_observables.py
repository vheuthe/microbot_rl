'''
This is for testing the observables for the transport case (mode 7)
'''
import json
import numpy as np
import matplotlib.pyplot as plt
from environments.rod import MD_ROD

def test_observables():
    '''
    Uses an initialized environment and shows all positions and
    orientations togetzer with the observables
    '''

    # Get an environment first
    env = get_environment()

    # Scatter both the rod and the target
    plt.scatter(env.rod[:,0], env.rod[:,1], color="blue")
    plt.scatter(env.target[:,0], env.target[:,1], color="grey")

    # Scatter the particles and quiver their orientations
    plt.scatter(env.particles[:,0], env.particles[:,1])
    plt.quiver(env.particles[:,0], env.particles[:,1], np.cos(env.particles[:,2]), np.sin(env.particles[:,2]))

    # Plot the observables as dots in front of each particle
    # in two rows for rod and target and in varying size for
    # observable "intensity"

    # Make dots in front of each particles
    dots_in_front = make_dots_in_front(env)

    # Plot the dots with sizes corresponding to the observables
    plt.scatter(np.real(dots_in_front.reshape([1,-1])), np.imag(dots_in_front.reshape([1,-1])), 70, color='white', edgecolors='black')
    for i in range(len(env.particles)):
        plt.scatter(
            np.real(dots_in_front[i,:]),
            np.imag(dots_in_front[i,:]),
            50 * env.obs[i,:], color="green")

    # Some cosmetics
    plt.axis('square')


def get_environment():
    '''
    Initializes an environment in a certain configuration
    and determines observables for testing them
    '''

    # Import the parameters first
    with open('./files_for_testing/parameters.json', 'r') as reader:
        parameters = json.load(reader)
    parameters['mode'] = 7
    parameters['cones'] = 10
    parameters['cone_angle'] = 2 * np.pi

    # Initialize the environment
    env = MD_ROD(**parameters)

    # Make up a configuration
    env = make_configuration(env)

    # Get the observables
    env.obs, _ = env.get_obs_rewards()

    return env


def make_configuration(env):
    '''
    Changes the configuration of the environment
    to an artificial one
    '''

    # The rod is centered at (0|0) and upright
    rod = np.linspace(- 1j * env.l_rod/2, 1j * env.l_rod/2, env.n_rod)

    # The target is parallel to the new rod, just moved 30 to the right
    target = rod + 30

    # There are two particles looking toward both rod and target from the left
    particles = np.array([[-40, 30, np.pi * 7/8], [-40, -30, np.pi * 3/4]])

    # Give everything to the environment
    env.rod = np.concatenate((np.real(rod).reshape((-1,1)), np.imag(rod).reshape((-1,1))), axis=1)
    env.target = np.concatenate((np.real(target).reshape((-1,1)), np.imag(target).reshape((-1,1))), axis=1)
    env.particles = particles

    # Don't forget the particle number
    env.N = 2

    return env


def make_dots_in_front(env):
    '''
    Makes dots in front of each particle
    in order to scatter the observables there
    '''

    # Preassign a dots array
    dots_in_front = \
               np.zeros((len(env.particles), 3 * env.cones)) \
        + 1j * np.zeros((len(env.particles), 3 * env.cones))

    # Give the dots positions
    for i in range(len(env.particles)):

        # Make up positions in front of the particle
        # "cones" many for both kinds of rod in lines
        # perpendicular to the particles orientation
        mid_of_dots_close = (env.particles[i,0] + 1j * env.particles[i,1]) \
            + 10 * np.exp(1j * env.particles[i,2])
        mid_of_dots_middle = (env.particles[i,0] + 1j * env.particles[i,1]) \
            + 15 * np.exp(1j * env.particles[i,2])
        mid_of_dots_far = (env.particles[i,0] + 1j * env.particles[i,1]) \
            + 20 * np.exp(1j * env.particles[i,2])

        # Dots have a spacing of 5
        dots_close = np.linspace(
            mid_of_dots_close - env.cones/2 * 5 * np.exp(1j * (env.particles[i,2] + np.pi/2)),
            mid_of_dots_close + env.cones/2 * 5 * np.exp(1j * (env.particles[i,2] + np.pi/2)),
            env.cones)
        dots_middle = np.linspace(
            mid_of_dots_middle - env.cones/2 * 5 * np.exp(1j * (env.particles[i,2] + np.pi/2)),
            mid_of_dots_middle + env.cones/2 * 5 * np.exp(1j * (env.particles[i,2] + np.pi/2)),
            env.cones)
        dots_far = np.linspace(
            mid_of_dots_far - env.cones/2 * 5 * np.exp(1j * (env.particles[i,2] + np.pi/2)),
            mid_of_dots_far + env.cones/2 * 5 * np.exp(1j * (env.particles[i,2] + np.pi/2)),
            env.cones)

        # Concatenate the dots
        dots_in_front[i,:] = np.concatenate((dots_close, dots_middle, dots_far))

    return dots_in_front


if __name__ == "__main__":

    test_observables()
