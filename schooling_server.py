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
import evolve_fortran_discreteFood as evolve

# DEFAULT CONFIG
parameters = {
  'max_particles': 200,
  'host_address': ('localhost', 22009),
  'training_frequency': 12,
  'training_epochs': 30,
  'reward_ratio': 0.5,
  'touch_penalty': 3,
  'food_max_x': 50,
  'food_max_y': 50,
  'food_amount': 666,
  'food_radius': 25,
  'input_dim': 15,
  'output_dim': 4,
  'lrPI': 0.003,
  'lrV': 0.003,
  'gamma': 0.99,
  'CL': 0.15,
  'en_coeff': 0.0,
  'lam': 0.98,
  'target_kl': 0.02,
  'save_models': True,
  'restart_models': False
  }

def get_observables_rewards(x, y, theta, food_x=0.0, food_y=0.0, food_amount=0.0, food_radius=1.0, reward_ratio=0.5, touch_penalty=3.0, input_dim=15, **unused):
  '''Wrapper around the fortran calculation for observables and reward'''
  return evolve.get_o_r_food_task(x, y, theta, 1, np.pi, 0, reward_ratio,
    touch_penalty, food_x, food_y, food_amount, 2*food_radius, input_dim, len(x))


# SERVER

def serve(parameters):

  # create RL Agent
  rl = AgentActiveMatter(**parameters)

  # update parameters in case a loaded model changed the dimensions
  parameters['input_dim'] = rl.input_dim
  parameters['output_dim'] = rl.n_actions

  #initialize food
  food_x = random.uniform(-parameters['food_max_x'], parameters['food_max_x'])
  food_y = random.uniform(-parameters['food_max_y'], parameters['food_max_y'])
  food_current = parameters['food_amount']

  # create TCP socket for communication
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(tuple(parameters['host_address'])) # field might be a list due to json parsing
  sock.listen(1)
  print("listening on {}:{}".format(*(parameters['host_address'])))

  # wait for matlab to connect
  connection, client_address = sock.accept()
  print("Client connected from {}:{}".format(*client_address))

  try:
    for update in itertools.count():

      # wait for matlab to send data
      data = connection.recv(8 * 4 * parameters['max_particles'])

      if data:
        # cast bytestream to double array and reshape to [x y theta state]
        data = np.array(struct.unpack(str(len(data)//8)+"d", data)).reshape((-1, 4))

        # parse data
        x = data[:,0]
        y = data[:,1]
        theta = data[:,2]
        theta[np.isnan(theta)] = 0
        lost = np.isnan(x)
        inboundary = (data[:,3] < 0)

        # debug
        print("Received data for action {:>4}: {:>3} particles, {:>3} lost, {:>3} in boundary condition, {:>3} valid."
          .format(update, len(x), np.sum(lost), np.sum(inboundary), sum(~np.logical_or(lost, inboundary))))

        # calculate observables and reward
        # TODO food is for now just a constant quantity at constant position
        obs, rew, eaten = get_observables_rewards(x[~lost], y[~lost], theta[~lost], food_x, food_y, **parameters)
        food_current -= eaten

        if food_current <= 0:
          food_x = random.uniform(-parameters['food_max_x'], parameters['food_max_x'])
          food_y = random.uniform(-parameters['food_max_y'], parameters['food_max_y'])
          food_current = parameters['food_amount']

        # remove invalid observables
        obs = obs[~inboundary[~lost],:]
        rew = rew[~inboundary[~lost]]
        invalid = np.argwhere(np.logical_or(lost, inboundary)).flatten().tolist()

        # feed data to RL network
        if update == 0:
          rl.initialize(obs)
        elif update % parameters['training_frequency'] == 0:
          rl.add_env_timeframe(invalid, obs, rew, False)
          rl.train_step(parameters['training_epochs'])
          rl.initialize(obs)
          print("Training network ...")
        else:
          rl.add_env_timeframe(invalid, obs, rew, False)

        # get actions and probabilitoes
        actions, pi_logp_all = rl.get_actions(True)
        # ensure that actions is column vector
        actions = actions.reshape((-1,1))
        # check number of actions
        if rl.n_actions == 3:
          actions = actions + 1
          pi_logp_all = np.append(np.full(actions.shape, -np.inf), pi_logp_all, axis=1)
        elif rl.n_actions != 4:
          raise NotImplementedError('Unsupported output_dim')
        # add food info as first row and flatten in 'Fortran' style
        data = np.append(
          [[food_x, food_y, parameters['food_radius'], parameters['food_amount'], food_current]],
          np.append(actions.reshape((-1,1)), pi_logp_all, axis=1),
          axis=0).flatten('F')
        # and send them (as bytestream)
        connection.sendall(struct.pack(str(len(data))+"d", *data))

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


