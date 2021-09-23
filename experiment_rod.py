import socket
import sys
import os
import json
import traceback
import struct
import itertools
import numpy as np
import random

from firstrl import AgentActiveMatter
from environments.rod import MD_ROD
import learning_rod


def serve_experiment():

    # load parameters provided by matlab:
    with open("./parameters.json") as paramfile:
        exp_parameters = json.load(paramfile)

    # if a model was provided, use corresponding parameters:
    if 'load_models' in exp_parameters and exp_parameters['load_models']:
        with open(exp_parameters['load_models'][:-5] + "parameters.json") as paramfile:
            parameters = json.load(paramfile)
    else:
        parameters = learning_rod.default_parameters

    # specific experimental parameters have higher precedence
    parameters.update(exp_parameters)

    # dump final configuration
    with open("./parameters.json", 'w', encoding='utf-8') as paramfile:
        json.dump(parameters, paramfile, ensure_ascii=False, indent=4)

    # --------------------------------------------------------------------------

    agent = AgentActiveMatter(**parameters)
    environment = MD_ROD(**parameters)

    # create TCP socket for communication
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(tuple(parameters['host_address'])) # field might be a list due to json parsing
    sock.listen(1)
    print("listening on {}:{}".format(*(parameters['host_address'])))

    # wait for matlab to connect
    connection, client_address = sock.accept()
    print("Client connected from {}:{}".format(*client_address))

    # --------------------------------------------------------------------------

    try:
        for update in itertools.count():

            # wait for matlab to send data
            data = connection.recv(8 * 4 * parameters['max_particles'])

            if data:
                # cast bytestream to double array and reshape to [x y theta state]
                data = np.array(struct.unpack(str(len(data)//8)+"d", data)).reshape((-1, 4))

                # There are maybe trailing zeros in both particles and rod
                particles = np.nan_to_num(data[np.logical_and(data[:,0] != 0, data[:,1] != 0, data[:,2] != 0, data[:,3] != 0), 0:3])
                rod = np.nan_to_num(data[np.logical_and(data[:,4] != 0, data[:,5] != 0), 0:3])

                # where x is NaN, particles are lost
                lost = np.isnan(data[np.logical_and(data[:,0] != 0, data[:,1] != 0, data[:,2] != 0, data[:,3] != 0), 0:3])

                # where state is negative
                inboundary = (particles[:,3] < 0)

                # debug
                print("Received data for action {:>4}: {:>3} particles, {:>3} lost, {:>3} in boundary condition, {:>3} valid."
                    .format(update, particles.shape[0], np.sum(lost), np.sum(inboundary), sum(~np.logical_or(lost, inboundary))))

                # calculate observables and rewards
                observables, rewards = environment.update(particles, environment.actions, rod, lost, update)

                # remove invalid observables
                observables = observables[~inboundary[~lost],:]
                rewards = rewards[~inboundary[~lost]]
                invalid = np.argwhere(lost | inboundary).flatten().tolist()

                # feed data to RL network
                if update == 0:
                    agent.initialize(observables)
                    values = np.zeros(len(rewards))
                elif update % parameters['train_pause'] == 0:
                    values = agent.add_environment_response(invalid, observables, rewards)
                    agent.train_step(parameters['training_epochs'])
                    agent.initialize(observables)
                    print("Training network ...")
                else:
                    values = agent.add_environment_response(invalid, observables, rewards)

                # get actions and probabilitoes
                environment.actions, logp = agent.get_actions()
                # ensure that actions is column vector
                environment.actions = np.array(environment.actions).reshape((-1,1))
                # check number of actions
                if agent.n_actions == 3:
                    environment.actions = environment.actions + 1
                    logp = np.append(np.full(environment.actions.shape, -np.inf), logp, axis=1)
                elif agent.n_actions != 4:
                    raise NotImplementedError('Unsupported n_actions')

                # Flatten data in 'Fortran' style
                data = np.concatenate((environment.actions, logp, np.array(rewards).reshape((-1,1)), np.array(values).reshape((-1,1))), axis=1).flatten('F')

                # and send them (as bytestream)
                connection.sendall(struct.pack(str(len(data))+"d", *data))

            else:
                print("System call interrupted, stopping server, saving models")
                agent.save_models('./model')
                break

    finally:
        connection.close()






if __name__ == "__main__":
    try:
        serve_experiment()
    except:
        traceback.print_exc(file=sys.stdout)
