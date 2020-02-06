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
  'training_epochs': 30,
  'input_dim': 5,
  'output_dim': 4,
  'lrPI': 0.003,
  'lrV': 0.003,
  'gamma': 0.99,
  'CL': 0.15,
  'en_coeff': 0.0,
  'lam': 0.98,
  'target_kl': 0.02,
  'save_models': False,
  'restart_models': False
  }


# ADDITIONAL ADDITIONAL FUNCTIONS

def parse_input(inputdata, input_dim):
  '''
  Takes a 1D array of shape (*, input_dim + 1)
  '''
  # reshape to expected columns (input_dim observables + 1 reward)
  inputdata = np.array(inputdata).reshape((-1, input_dim + 1))
  # get nan lines as 1D list
  lost = np.argwhere(np.isnan(inputdata)[:, 0]).flatten().tolist();

  if len(np.argwhere(np.isnan(inputdata))) > 0:
    print(inputdata)

  # and remove them (reshape needed as logic indexing flattens the matrix)
  inputdata = np.reshape(inputdata[~np.isnan(inputdata)], (-1, input_dim + 1))
  # return lost, obs, rewards
  return lost, inputdata[:, 0:input_dim], inputdata[:, input_dim:input_dim+1]


# SERVER

def serve(parameters):

  # create RL Agent
  rl = AgentActiveMatter(**parameters)

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

        # parse 1d array
        lost, obs, rewards = parse_input(data, parameters['input_dim'])

        # feed data to RL network
        if frame == 0:
          rl.initialize(obs)
        elif frame % parameters['training_frequency'] == 0:
          rl.add_env_timeframe(lost, obs, rewards, False)
          rl.train_step(parameters['training_epochs'])
          rl.initialize(obs)
        else:
          rl.add_env_timeframe(lost, obs, rewards, False)

        # get actions
        actions = rl.get_actions()
        # and send them (as bytestream)
        connection.sendall(struct.pack(str(len(actions))+"d", *actions))

      else:
        print("System call interrupted, Stopping Server")

        if parameters['save_models'] :
          print('Saving models to ' + parameters['models_savepath'])
          rl.save_models(parameters['models_savepath'], True)

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


