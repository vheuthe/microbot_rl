import socket
import sys
import json
import traceback
import struct
import itertools
import time
import h5py
import numpy as np

from firstrl import AgentActiveMatter
from environments.rod import MD_ROD
import learning_rod


def serve_experiment():
    '''
    This communicates with the experiment, computes rewards and
    observables for it. It mainly uses environments/rod.update
    for that.
    '''

    # load parameters provided by matlab:
    with open("./parameters.json") as paramfile:
        exp_parameters = json.load(paramfile)

    # if a model was provided, use corresponding parameters:
    if 'load_models' in exp_parameters and exp_parameters['load_models']:
        with open(exp_parameters['load_models'][:-5] + "parameters.json") as paramfile:
            parameters = learning_rod.default_parameters.copy()
            parameters.update(json.load(paramfile))
    else:
        parameters = learning_rod.default_parameters

    # specific experimental parameters have highest priority
    parameters.update(exp_parameters)

    # Infer some parameters
    parameters['n_obs'] = 3 * parameters['cones']

    # Make up an episode length (need to be poisson distributed somehow)
    if parameters["episodic"]:
        n_step_ep = 3 * np.random.poisson(lam=np.round(1/(1-parameters["gamma"])), size=1)
    else:
        n_step_ep = parameters["train_frames"]
    parameters['n_step_ep'] = int(n_step_ep)

    # dump final configuration
    with open("./parameters.json", 'w', encoding='utf-8') as paramfile:
        json.dump(parameters, paramfile, ensure_ascii=False, indent=4)

    # Set up the data storage file in h5 format for storing the re-simulated data
    store_file = h5py.File('./resimulated_data.h5', 'w')

    # --------------------------------------------------------------------------

    # Initiate the agent
    agent = AgentActiveMatter(**parameters)

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
            data = connection.recv(8)

            # Time the execution of one iteration
            time_beginning = time.perf_counter()

            if data:
                n_data = struct.unpack('I', data)[0]
                data = connection.recv(8 * n_data)
                receive_num = 1
                # cast bytestream to double array and reshape to [x y theta state]
                data_unpacked = np.array(struct.unpack(str(len(data)//8)+"d", data)).reshape((-1, 1))
                print(f"data_unpacked has the shape {data_unpacked.shape} after receive number {receive_num}")

                while len(data_unpacked) < n_data:
                    data = connection.recv(8 * n_data)
                    receive_num += 1
                    data_unpacked = np.append(data_unpacked,
                        np.array(struct.unpack(str(len(data)//8)+"d", data)).reshape((-1, 1)))
                    print(f"data_unpacked has the shape {data_unpacked.shape} after receive number {receive_num}")

                # When all the data is received, it is brought into the right shape
                data_reshaped = data_unpacked.reshape((-1,7))

                # There are maybe trailing zeros in both particles and rod
                is_particle_data = np.logical_or(
                    data_reshaped[:,0] != 0,
                    data_reshaped[:,1] != 0,
                    data_reshaped[:,2] != 0
                    )
                is_rod_data = np.logical_or(
                    data_reshaped[:,4] != 0,
                    data_reshaped[:,5] != 0
                    )
                is_frame_data = data_reshaped[:,6] != 0

                # Take a look at the frame first to be able to tag all other data
                frame = data_reshaped[is_frame_data, 6]
                particles = np.nan_to_num(data_reshaped[is_particle_data, 0:3])
                rod = np.nan_to_num(data_reshaped[is_rod_data, 4:6])

                # In the first update, initiate the environment and send the target position
                if update == 0:
                    environment = MD_ROD(**parameters)
                    environment.rod = rod
                    environment.target = environment.make_target(rod)
                    data_send = environment.target.flatten('F')
                    connection.sendall(struct.pack(str(len(data_send))+"d", *data_send))
                    print("Sent the target position")

                # The actions are already the old actions!
                # (since the new actions are not known yet)
                old_actions = np.nan_to_num(data_reshaped[is_particle_data, 3]) - 1

                # Where x is NaN, particles are lost
                lost = np.any(np.isnan(data_reshaped[is_particle_data, 0:3]), axis=1)

                # Where state is negative, the particles are in the border-region
                inboundary = old_actions < 0

                # Debug
                print("Received data for action {:>4}: {:>3} particles, {:>3} lost, {:>3} in boundary condition, {:>3} valid."
                    .format(update, particles.shape[0], np.sum(lost), np.sum(inboundary), sum(~np.logical_or(lost, inboundary))))

                # Copy old_part so I can save them later
                # (has to be done before environment.update,
                # because old_part is altered there already)
                if not update == 0:
                    old_old_part = environment.old_part

                # calculate observables and rewards
                observables_raw, rewards_raw, found = environment.update(particles, old_actions, rod, lost, update)

                # Is this the final step of the episode? If so, train a last time and end the episode
                final = (parameters["episodic"] and environment.task_achieved) or update == n_step_ep

                # remove invalid observables
                observables = observables_raw[~(inboundary | lost),:]
                rewards = rewards_raw[~(inboundary | lost)]
                invalid = np.argwhere(lost | inboundary).flatten().tolist()

                # This is a bad fix, but no time to find the bug (22.12.21)
                observables[np.isnan(observables)] = 0
                rewards[np.isnan(rewards)] = 0

                # feed data to RL network
                if update == 0:
                    agent.initialize(observables)
                    values = np.zeros(len(rewards))

                elif update % parameters['train_pause'] == 0 or final:

                    print("Training network ...")

                    values = agent.add_environment_response(invalid, observables, rewards)
                    agent.train_step()
                    agent.initialize(observables)

                    # For post-training evaluation, the weights are saved after each update
                    agent.save_weights('./model', update)

                else:
                    values = agent.add_environment_response(invalid, observables, rewards)

                # get actions and probabilitoes
                actions, logp = agent.get_actions()
                # ensure that actions is column vector
                actions = np.array(actions).reshape((-1,1))
                # check number of actions
                if agent.n_actions == 3:
                    actions = actions + 1
                    logp = np.append(np.full(actions.shape, -np.inf), logp, axis=1)
                elif agent.n_actions != 4:
                    raise NotImplementedError('Unsupported n_actions')

                # For debugging send back the current frame, too,
                # together with whether or not the episode stops
                frame_data = np.full_like(actions, 0)
                frame_data[0] = frame
                frame_data[1] = final

                # Flatten data in 'Fortran' style
                data_send = np.concatenate((actions, logp, np.array(rewards).reshape((-1,1)), np.array(values).reshape((-1,1)), frame_data), axis=1).flatten('F')

                # and send them (as bytestream)
                connection.sendall(struct.pack(str(len(data_send))+"d", *data_send))
                print("Sent data for update {} with length {}".format(update, data_send.shape[0]))

                # Saving the hypothetical particles, the hypothetical rod, the old
                # actions, the old particles and the frame for later investigation
                # (has to be done after environment.update)
                if parameters["rew_mode"] == "WLU" or parameters["rew_mode"] == "WLU_experiment":
                    rod_name = f"update{update}/hyp_rod"
                    parts_name = f"update{update}/hyp_parts"
                    frame_name = f"update{update}/frame"
                    store_file.create_dataset(rod_name, compression='gzip', data=environment.hyp_rod)
                    store_file.create_dataset(parts_name, compression='gzip', data=environment.hyp_parts)
                    store_file.create_dataset(frame_name, compression='gzip', data=frame)
                    if not update == 0:
                        old_parts_name = f"update{update}/old_parts"
                        old_actions_name = f"update{update}/old_actions"
                        store_file.create_dataset(old_parts_name, compression='gzip', data=old_old_part[~lost[~found],:])
                        store_file.create_dataset(old_actions_name, compression='gzip', data=environment.old_actions)

                print(f"Execution took {time.perf_counter() - time_beginning} seconds")

                # End the episode, if time ran out or the task is achieved
                if final:
                    if environment.task_achieved:
                        print("The task was achieved, ending episode")
                    elif update == n_step_ep:
                        print("Time ran out, ending episode")
                    else:
                        print("Neither time ran out nor the task was achieved")
                    agent.save_models('./model')
                    store_file.close()
                    break

            else:
                print("System call interrupted, stopping server, saving models")
                agent.save_models('./model')
                store_file.close()
                break

    finally:
        connection.close()






if __name__ == "__main__":
    try:
        serve_experiment()
    except:
        traceback.print_exc(file=sys.stdout)
