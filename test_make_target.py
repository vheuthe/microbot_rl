'''
For testing the make_target method of environments.rod
'''
import json
import matplotlib.pyplot as plt
from environments.rod import MD_ROD

def test_make_target(number):
    '''
    Plots a lot of randomly initialized targets to test the
    make_target method of environments.rod
    '''

    # Import the parameters first (make sure to be in mode 7, target transport)
    with open('./files_for_testing/parameters.json', 'r') as reader:
        parameters = json.load(reader)
    parameters["mode"] = 7

    # Do a loop, generate a target every time and plot it all together
    for _ in range(number):

        # Initialize the environment
        environment = MD_ROD(**parameters)

        # Plot the target
        plt.scatter(environment.target[:,0], environment.target[:,1], 10, 'k')

    plt.axis('square')


if __name__ == "__main__":
    test_make_target(100)
