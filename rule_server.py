import socket
import sys
import os
import json
import traceback
import struct
import itertools
import numpy as np
from firstrl import AgentActiveMatter

# DEFAULT CONFIG
parameters = {
  'max_particles': 200,
  'host_address': ('localhost', 22009),
  'training_frequency': 12,
  'training_epochs': 50,
  'input_dim': 5,
  'output_dim': 4,
  'lrPI': 0.003,
  'lrV': 0.003,
  'gamma': 0.97,
  'CL': 0.07,
  'en_coeff': 0.0,
  'lam': 0.97,
  'target_kl': 0.02,
  'save_models': True,
  'models_savepath': './model',
  'load_models': None
  }


# ADDITIONAL ADDITIONAL FUNCTIONS

def parse_input(inputdata, input_dim):
  '''
  Takes a 1D array of shape (*, input_dim + 1)
  '''
  # reshape to expected columns (input_dim observables + 1 reward)
  inputdata = np.array(inputdata).reshape((-1, input_dim + 1))
  # get nan lines as 1D list
  lost = np.argwhere(np.isnan(inputdata)[:, 0]).flatten().tolist()

  # and remove them (reshape needed as logic indexing flattens the matrix)
  inputdata = np.reshape(inputdata[~np.isnan(inputdata)], (-1, input_dim + 1))
  # return lost, obs, rewards
  return lost, inputdata[:, 0:input_dim], inputdata[:, input_dim:input_dim+1]


# SERVER

def serve(parameters):

  # create RL Agent
  agent = AgentActiveMatter(**parameters)

  # create TCP socket for communication
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(tuple(parameters['host_address'])) # field might be a list due to json parsing
  sock.listen(1)
  print("listening on {}:{}".format(*(parameters['host_address'])))

  # wait for matlab to connect
  connection, client_address = sock.accept()
  print("Client connected from {}:{}".format(*client_address))

  try:
    for frame in itertools.count():

      # wait for matlab to send data
      data = connection.recv(8 * parameters['max_particles'] * (parameters['input_dim'] + 1))

      if data:
        # cast bytestream to double array
        data = np.array(struct.unpack(str(len(data)//8)+"d", data))
        print("Received data for action ", frame)

        # parse 1d array
        lost, obs, rewards = parse_input(data, parameters['input_dim'])
        print(len(rewards), " valid particles, ", len(lost), " lost")

        # feed data to RL network
        if frame == 0:
          agent.initialize(obs)
        elif frame % parameters['training_frequency'] == 0:
          agent.add_environment_response(lost, obs, rewards)
          agent.train_step(parameters['training_epochs'])
          agent.initialize(obs)
          print("Training network ...")
        else:
          agent.add_environment_response(lost, obs, rewards)

        # get actions and probabilitoes
        actions, logp = agent.get_actions()
        # flatten in 'Fortran' style
        data = np.append(actions.flatten('F'), logp.flatten('F'))
        # and send them (as bytestream)
        connection.sendall(struct.pack(str(len(data))+"d", *data))

      else:
        print("System call interrupted, Stopping Server")

        if parameters['save_models'] :
          print('Saving models to ' + parameters['models_savepath'])
          agent.save_models(parameters['models_savepath'])

        break

  finally:
      connection.close()


if __name__ == "__main__":
  try:
    if os.path.isfile("./rl-parameters.json"):
      with open("./rl-parameters.json") as paramfile:
        parameters.update(json.load(paramfile))
    with open("./rl-parameters.json", 'w', encoding='utf-8') as paramfile:
      json.dump(parameters, paramfile, ensure_ascii=False, indent=4)
    serve(parameters)
  except:
    traceback.print_exc(file=sys.stdout)


