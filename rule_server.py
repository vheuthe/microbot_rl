import socket
import struct
import itertools
import numpy as np
import math
from firstrl import AgentActiveMatter 

# CONFIG

address = ('localhost', 22009)
maxN    = 200;

train_freq = 10

parameters = {}


# ADDITIONAL ADDITIONAL FUNCTIONS

def parse_input(inputdata):
  '''
  Takes a 1D array of shape (*, 3)
  '''
  lost = []
  inputdata = np.reshape(np.array(inputdata),(-1, 3))
  lost = np.reshape(np.argwhere(np.isnan(inputdata)[:, 0]), (-1,))
  mask = ~np.isnan(inputdata)
  inputdata = inputdata[mask]
  lost = [index for index in lost]
  return lost, np.reshape(inputdata,(-1,3))


def get_obs_rewards(pos):
  N = pos.shape[0]
  # change here dimension of observables
  obs = np.zeros((N,5))  # each particle has 5 slices of cone of sight
  rewards = np.zeros((N,1))
  value_cone=np.array((1.0, 1.0, 1.0, 1.0, 1.0))
  for i in range(N):
    for j in range(N):
      if i!=j:
        # here calculates distance between particle and relative orientation 
        # respective to the first particle
        dist, rel_theta = get_dist_reltheta(pos[i], pos[j])
        # from here it is arbitrary rewards and observables
        n_cone = math.floor((rel_theta/math.pi + 1.0)%(2.0)-0.5)
        if n_cone > -1 and n_cone < 5: 
          #if dist < 15: 
          rewards[i]     += 2/(dist/5+5)*value_cone[n_cone]
          obs[i][n_cone] += 2/(dist/5+5)
        if (obs[i] == 0).all(): rewards[i] -= 2
                    #HERE I SHOULD USE A SATURATING VALUE OF SOMETHING. PERHAPS THE SAME AS IN CLEMEN'S WORK
  return obs, rewards
      
def get_dist_reltheta(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    dist = math.sqrt(dx*dx + dy*dy)
    rel_theta = math.atan2(dy, dx) - A[2]
    return dist, rel_theta


# SERVER

def serve():

        rl = AgentActiveMatter(**parameters)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(address)
        sock.listen(1)

        print("listening on {}:{}".format(*(address)))

        connection, client_address = sock.accept()

        print("Client connected from {}:{}".format(*client_address))

        try:
            for frame in itertools.count():
                data = connection.recv(8*maxN)
                if data:
                        
                    data = np.array(struct.unpack(str(len(data)//8)+"d", data))

                    lost, pos = parse_input(data)
                    obs, rewards = get_obs_rewards(pos)

                    if frame == 0:
                        rl.initialize(obs)
                    elif frame % train_freq == 0:
                        rl.add_env_timeframe(lost, obs, rewards)
                        rl.train_step()
                        rl.initialize(obs)
                    else:
                        rl.add_env_timeframe(lost, obs, rewards)

                    actions = rl.get_actions()

                    connection.sendall(struct.pack(str(len(actions))+"d", *actions))
                    
                    print("Recived some data and replied")
 
                else:
                    print("System call interrupted, Stopping Server")
                    break
        finally:
            connection.close()


if __name__ == "__main__":
    serve()


