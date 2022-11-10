'''
This is for testing the difference rewards (previously called WLU)
for the transport case (mode 7)
'''
import json
import numpy as np
import matplotlib.pyplot as plt
from environments.rod import MD_ROD


def make_scenario(number):
    '''
    Making different scenarios in which to test the reward function
    '''

    # Import the parameters first
    with open('./files_for_testing/parameters.json', 'r') as reader:
        parameters = json.load(reader)
    parameters['mode'] = 7

    # Device a test scenario
    environment = MD_ROD(**parameters)

    if number == 1:
        # Rod is parallely pushed towards target

        # The new rod is centered at (0|0) and upright
        rod = np.linspace(- 1j * environment.l_rod/2, 1j * environment.l_rod/2, environment.n_rod)

        # The target is parallel to the new rod, just moved 30 to the right
        target = np.linspace(30 - 1j * environment.l_rod/2, 30 + 1j * environment.l_rod/2, environment.n_rod)

        # The old rod was 1 more to the left than the new rod
        old_rod = rod - 1

    if number == 2:
        # Rod is parallely pushed away from the target

        # The new rod is centered at (0|0) and upright
        rod = np.linspace(- 1j * environment.l_rod/2, 1j * environment.l_rod/2, environment.n_rod)

        # The target is parallel to the new rod, just moved 30 to the right
        target = np.linspace(30 - 1j * environment.l_rod/2, 30 + 1j * environment.l_rod/2, environment.n_rod)

        # The old rod was 1 more to the left than the new rod
        old_rod = rod + 1

    if number == 3:
        # Rod is parallely pushed towards the target (with smaller distance)

        # The new rod is centered at (0|0) and upright
        rod = np.linspace(- 1j * environment.l_rod/2, 1j * environment.l_rod/2, environment.n_rod)

        # The target is parallel to the new rod, just moved 15 to the right
        target = np.linspace(15 - 1j * environment.l_rod/2, 15 + 1j * environment.l_rod/2, environment.n_rod)

        # The old rod was 1 more to the left than the new rod
        old_rod = rod - 1

    if number == 4:
        # Rod is parallely pushed towards the target (with even smaller distance)

        # The new rod is centered at (0|0) and upright
        rod = np.linspace(- 1j * environment.l_rod/2, 1j * environment.l_rod/2, environment.n_rod)

        # The target is parallel to the new rod, just moved 2 to the right
        target = np.linspace(2 - 1j * environment.l_rod/2, 2 + 1j * environment.l_rod/2, environment.n_rod)

        # The old rod was 1 more to the left than the new rod
        old_rod = rod - 1

    elif number == 5:
        # Rod is perpendicular to and then rotated towards target clockwise

        # The target is horizontal at (0|0)
        target = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The old rod is vartical with a little gap to the target at
        old_rod = np.linspace(1j * 15, 1j * (15 + environment.l_rod), environment.n_rod)

        # The new rod is tilted a bit clockwise
        rod = old_rod * np.exp(-1j * np.pi/20)

    elif number == 6:
        # Rod is perpendicular to and then rotated towards target counter-clockwise

        # The target is horizontal at (0|0)
        target = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The old rod is vartical with a little gap to the target at
        old_rod = np.linspace(1j * 15, 1j * (15 + environment.l_rod), environment.n_rod)

        # The new rod is tilted a bit counter-clockwise
        rod = old_rod * np.exp(1j * np.pi/20)

    elif number == 7:
        # Rod is staying at fixed distance to the target but rotated

        # The target is horizontal at (0|-50)
        target = np.linspace(-environment.l_rod/2 - 1j * 50, environment.l_rod/2 - 1j * 50, environment.n_rod)

        # The old rod is vertical at (0|0)
        old_rod = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The old rod is vertical at (0|0)
        rod = old_rod * np.exp(1j * np.pi/20)

    elif number == 8:
        # The rod is nearly in the target and rotated towards it

        # The target is horizontal at (0|0)
        target = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The rod is in the target
        rod = target

        # The old rod is the rod slightly rotated
        old_rod = rod * np.exp(1j * np.pi/100)

    elif number == 9:
        # Performance upon 90 deg flip into target

        # The target is horizontal at (0|0)
        target = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The rod is in the target
        rod = target

        # The old rod is the rod rotated by 90 deg
        old_rod = rod * np.exp(1j * np.pi/2)

    elif number == 10:
        # Performance upon 90 deg flip into target

        # The target is horizontal at (0|0)
        target = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The rod is in the target
        old_rod = target

        # The old rod is the rod rotated by 90 deg
        rod = old_rod * np.exp(1j * np.pi/2)

    elif number == 11:
        # Rod is rotated around one of its ends

        # The old rod is horizontal and ends at (0|0)
        old_rod = np.linspace(-environment.l_rod, 0, environment.n_rod)

        # The target is a little further down
        target = old_rod - 1j * 50

        # The rod is the old rod rotated around its right end away from the target
        rod = old_rod * np.exp(1j * -np.pi/20)

    elif number == 12:
        # Rod is rotated around one of its ends (but starting tilted)

        # The rod is horizontal and ends at (0|0)
        rod = np.linspace(-environment.l_rod, 0, environment.n_rod)

        # The target is a little further down
        target = rod - 1j * 50

        # The old rod is the rod rotated around its right end away from the target
        old_rod = rod * np.exp(1j * np.pi/20)

    elif number == 13:
        # Rod is far from the target, starts parallel and is rotated

        # The old rod is horizontal at (0|0)
        old_rod = np.linspace(-environment.l_rod/2, environment.l_rod/2, environment.n_rod)

        # The target is further down
        target = old_rod - 1j * 100

        # The rod is the old rod rotated a little bit
        rod = old_rod * np.exp(1j * np.pi/20)

    # Give everything to the environment
    environment.rod = np.concatenate((np.real(rod).reshape((-1,1)), np.imag(rod).reshape((-1,1))), axis=1)
    environment.target = np.concatenate((np.real(target).reshape((-1,1)), np.imag(target).reshape((-1,1))), axis=1)
    environment.old_rod = np.concatenate((np.real(old_rod).reshape((-1,1)), np.imag(old_rod).reshape((-1,1))), axis=1)

    return environment

def test_performance(numbers):
    '''
    Calls the different scenarions, plots them and determines the performance
    '''

    for num in numbers:

        # Get the scenario
        env = make_scenario(num)

        # Plot rod, old rod and target
        plt.figure()
        plt.scatter(env.rod[:,0], env.rod[:,1], color='black', alpha=0.3)
        plt.scatter(env.old_rod[:,0], env.old_rod[:,1], color='blue', alpha=0.3)
        plt.scatter(env.target[:,0], env.target[:,1], color='yellow', alpha=0.3)
        plt.axis('square')

        # Quiver all the arrows from new to old rod particles
        plt.quiver(
            env.old_rod[:,0],
            env.old_rod[:,1],
            env.rod[:,0] - env.old_rod[:,0],
            env.rod[:,1] - env.old_rod[:,1],
            scale=100, alpha=0.2)

        # Determine the performance
        performance = env.det_performance(env.rod)

        # Write the performance to the graph
        plt.text(10, 10, f"Performance = {performance}")


if __name__ == "__main__":
    test_performance([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])