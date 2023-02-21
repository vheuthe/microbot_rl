from environments.rod import MD_ROD
import learning_rod
import numpy as np
import matplotlib.pyplot as plt

def test_initialization():
    '''
    Initialize the environment and plot the particle positions and orientations
    '''

    # Get the default parameters first
    def_params = learning_rod.default_parameters

    # Select a bunch of initialization scalings
    scalings = [1, 2, 3, 4, 10, 20]

    # Initialize with each scaling and plot the particles position and orientation
    for scale in scalings:

        # Initialize
        def_params['start_dist_scale'] = scale
        env = MD_ROD(**def_params)

        # Plot
        plt.figure()
        scale_ax = plt.axes()
        scale_ax.scatter(env.particles[:,0], env.particles[:,1], color='black')
        scale_ax.scatter(env.rod[:,0], env.rod[:,1], color='green')
        scale_ax.quiver(env.particles[:,0], env.particles[:,1], np.cos(env.particles[:,2]), np.sin(env.particles[:,2]))
        scale_ax.axis('square')


if __name__ == "__main__":
    test_initialization()
