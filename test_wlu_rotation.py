'''
Testing the WLU rewards
'''
import json
import numpy as np
import matplotlib.pyplot as plt
from environments.rod import MD_ROD

def two_particles_test(rel_pos, n_reps):
    '''
    This iterates through particles positions along the rod,
    places two particles at each relative position and lets
    them push against the rod to determine their impact.
    '''

    # Import the parameters first
    with open('./files_for_testing/parameters.json', 'r') as reader:
        parameters = json.load(reader)

    # Allocate performance
    performance = np.zeros([len(rel_pos), n_reps])

    # Iterate through relative positions along rod and reps
    for i in range(len(rel_pos)):
        for j in range(n_reps):
            # Initialize the environment (has to be done every time for rod)
            environment = MD_ROD(**parameters)
            environment.fr_rod = 8

            # Select the right rel_pos
            pos = rel_pos[i]

            # Put artificial particles at the right place
            put_particles_at_their_places(environment, pos)

            # Let the simulation run one step and evaluate the performance
            environment.evolve_MD(environment.actions)
            performance[i, j] = environment.det_performance(environment.rod)

    # Average over reps
    performance_av = np.mean(performance, axis=1)

    # Plot the performance with respect to relative particle position
    plt.figure()
    plt.plot(rel_pos, performance_av)
    plt.xlabel('Pos. along rod /µm')
    plt.ylabel('$\Delta \\theta$')
    plt.title(f'fr_rod = {environment.fr_rod}')


def put_particles_at_their_places(environment, rel_pos, test_plot=False):
    '''
    This puts particles at the right positions with the right
    orientations to push against the rod and rotate it
    '''

    # Erase the old particles
    environment.particles = []

    # Get some basic properties first
    rod_comp = complex(np.mean(environment.rod[:,0]), np.mean(environment.rod[:,1]))
    rod_theta = np.angle(complex(environment.rod[-1,0] - environment.rod[0,0], environment.rod[-1,1] - environment.rod[0,1]))

    # Make up positions first
    part_comp = np.array([rod_comp + rel_pos * np.exp( 1j * rod_theta) + 6 * np.exp(1j * (rod_theta + np.pi/2)),
                 rod_comp + rel_pos * np.exp(-1j * rod_theta) + 6 * np.exp(1j * (rod_theta - np.pi/2))]).reshape(2, -1)
    environment.particles = np.append(np.real(part_comp), np.imag(part_comp), axis=1)

    # Then orientations
    environment.particles = np.append(environment.particles,
                                    np.array([rod_theta - np.pi/2,
                                              rod_theta + np.pi/2]).reshape(2, -1),
                                    axis=1)

    # And actions
    environment.actions = np.full_like(environment.particles[:,0], 1)

    # Change some additional things for sanity
    environment.old_part = np.zeros_like(environment.particles)
    environment.old_actions = np.zeros_like(environment.actions)
    environment.N = 2

    # Plot the situation for making sure it's right
    if test_plot:
        plt.figure()
        plt.scatter(environment.rod[:,0], environment.rod[:,1], 10, 'k')
        plt.scatter(environment.particles[:,0], environment.particles[:,1], 10, 'g')
        plt.quiver(
            environment.particles[:,0],
            environment.particles[:,1],
            np.cos(environment.particles[:,2]),
            np.sin(environment.particles[:,2]))
        plt.axis('square')

    return

if __name__ == "__main__":
    n_repss = 5
    rel_poss = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    two_particles_test(rel_poss, n_repss)