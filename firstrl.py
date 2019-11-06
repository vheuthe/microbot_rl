
import numpy as np
import tensorflow as tf
import scipy.signal

tf.keras.backend.set_floatx('float32')

# Ausiliary functions

def from_policy_to_actions(logp):
    '''
    takes distribution of log probabilities over discrete set of actions
    and gives out one randomly, after normalization
    MUST RETURN only one value: index of action!
    '''
    prob = np.exp(logp)
    prob = prob / np.sum(prob)
    n_action = prob.shape[1]
    action=np.random.choice(n_action,p=prob[0])
    return action

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# This is an ausiliary class that memorize the trajectory of a single particle

class SAM():
  '''
  SmartActiveMatter class, which stores individual history of single particles.
  contains:
   - all past observables (obs), actions (act), logp of actions (logp), rewards (rew), and values of state (val)
  instantaneous state is stored in self.current
  '''
  def __init__(self, obs):

    # check obs shape is (1, n_inputs)!
    self.current = obs
    self.obs = np.empty((0,obs.shape[1]))
    self.act = np.empty((0), dtype=np.int8)
    self.rew = []
    self.val = []
    self.logp = []

  def add_obs_rew_val(self, o, r, v):
    self.obs = np.append(self.obs, self.current, axis=0)
    self.current = o
    self.val = np.append(self.val, v)
    self.rew = np.append(self.rew, r)

  def add_act_logp(self, a, logp):
    self.act = np.append(self.act, a)
    self.logp = np.append(self.logp, logp)

# ----------------------------------
# ACTUAL REINFORCEMENT LEARNING CODE
# ----------------------------------

class AgentActiveMatter():
  '''
  Complete class for Reinforcement Learning.
  Contains two Neural Networks for Actor and Critic, tracks and stores the trajectories
  '''
  def __init__(self, input_dim=5, output_dim=3, lrPI=0.01, lrV=0.01, gamma=0.95, CL=0.03, en_coeff=0.0, lam=1.00, batch_size=32, target_kl=0.02, models_rootname='./model', restart_models = False, **unused_parameters):

    # internal knowledge
    self.input_dim = input_dim
    self.n_actions = output_dim
    self.optimizer = tf.optimizers.Adam(learning_rate=lrPI) # optimizer
    self.gamma = gamma                                      # gamma for discount future rewards
    self.lam = lam                                          # lambda for GAE
    self.CL = CL                                            # clipping parameter
    self.en_coeff = en_coeff                                # entropy coefficient
    self.lrPI = lrPI                                        # learning rate policy
    self.lrV = lrV                                          # learning rate value
    self.batch_size = batch_size                            # batch_size
    self.target_kl = target_kl                              # target KL divergence for update early stop
    self.N = None
    self.checkpointID = 0                                   # counter for model checkpoints

    self.particles = []
    self.reset_batch()                             # initialize memory (to zero)

    self.critic_path = models_rootname+'_critic/'
    self.policy_path = models_rootname+'_policy/'
  
    # ------------------------------------------
    if (restart_models):
      self.critic = tf.keras.models.load_model(self.critic_path)
      self.policy = tf.keras.models.load_model(self.policy_path)

      loaded_input_dim = self.critic.layers[0].input_shape[1]
      loaded_output_dim = self.policy.layers[-1].input_shape[1]
      
      assert (loaded_input_dim == self.input_dim), 'input dimension does not match with loaded model'
      assert (loaded_output_dim == self.n_actions), 'action dimension does not match with loaded model'

    else:
      # create actor
      self.policy = tf.keras.Sequential([
      # Adds a densely-connected layer:
      tf.keras.layers.Dense(16, activation='tanh', input_shape=(self.input_dim,)),
      # Add another dense layer:
      tf.keras.layers.Dense(8, activation='tanh'),
      # Add an output layer with n_actions output units:
      tf.keras.layers.Dense(self.n_actions, activation='linear')])
      # ------------------------------------------

      # ------------------------------------------
      # create critic
      self.critic = tf.keras.Sequential([
      # Adds a densely-connected layer with 6 units to the model:
      tf.keras.layers.Dense(16, activation='relu', input_shape=(self.input_dim,), kernel_initializer='zeros'),
      # Add another:
      tf.keras.layers.Dense(16, activation='relu'),
      # Add an output layer with 4 output units:
      tf.keras.layers.Dense(1, activation='linear')])
      # ------------------------------------------
      self.critic.compile(optimizer=tf.optimizers.Adam(learning_rate=lrV), loss='mse')
      # ------------------------------------------

#  -----------------------------
  def save_models(self, final_save = False):
    '''
    Saves critic and policy models in tf format at position defined by models_rootname + '_critic/' or '_policy'
    '''
    if (final_save):
      tf.keras.models.save_model(self.critic, self.critic_path)
      tf.keras.models.save_model(self.policy, self.policy_path)
    else:
      cpath = self.critic_path+'checkpoints/ckpt-'+str(self.checkpointID)
      ppath = self.policy_path+'checkpoints/ckpt-'+str(self.checkpointID)
      self.critic.save_weights(cpath)
      self.policy.save_weights(ppath)
      self.checkpointID += 1


  def reset_batch(self):
    '''
    set batch of memory to null
    '''
    self.obs = np.empty((0,self.input_dim))
    self.logp = np.empty((0))
    self.adv = np.empty((0))
    self.target = np.empty((0))
    self.actions = np.empty((0), dtype=np.int8)
    self.idx_start = 0
    self.idx_now = 0

  def initialize(self, obs):
    '''
    get first observables for all paricles
    obs MUST be of shape (N_particles, input_dim)
    '''
    self.particles = [SAM(o.reshape(1,self.input_dim)) for o in obs]    # creates list of SAM objects, where to store individual particles
    self.N = obs.shape[0]

  def add_env_timeframe(self, lost, new_obs, rewards):
    '''
    receives information about the present time step
    from the outside world:
    - lost is list of IDs of particles that have been lost in tracking !! AS NUMBERED IN THE PREVIOUS FRAME!!
    - new_obs is a numpy array of observables of all particles (N_particle, n_input_dim)
    - rewards is a numpy array defining the rewards of all particles (N_particle)
    '''

    for ID_lost in sorted(lost, reverse=True):
      self.finish_path(True, ID_lost)

    # self.N -= len(lost) can subtract too much at the moment

    for i, (o, r) in enumerate(zip(new_obs, rewards)):
      o = o.reshape(1,self.input_dim)
      # if (i < self.N):
      if i < len(self.particles): # should work as well?
        par = self.particles[i]
        v = self.critic(par.current).numpy()[0,0]
        par.add_obs_rew_val(o, r, v)
      else:
        self.add_in_memory(o)

    # new number of particles
    self.N = new_obs.shape[0]

  def add_in_memory(self, o):
    self.particles.append(SAM(o))

  def get_actions(self):
    '''
    returns numpy array of actions
    to take given the current observables
    and updates the particles with action, current value and logp
    '''
    actions = []

    for par in self.particles:
      # for each particle,
      # its current observable is used to choose an action
      o = par.current
      pi_logits = self.policy(o)
      pi_logp = tf.nn.log_softmax(pi_logits)
      a = from_policy_to_actions(pi_logp)
      logp = pi_logp[0, a]
      par.add_act_logp(a, logp)
      actions = np.append(actions, a)

    return actions

  def finish_path(self, lost = False, ID = -1):
    """
    - FROM SPINNING UP's PPO CODE -
    Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending. This looks back in the buffer to where the
    trajectory started, and uses rewards and value estimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.
    The "last_val" argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    ----
    In this realization, each particle is considered as an individual
    trajectory.
    If the particle is lost, it is considered as it received a "done"
    signal, and the last action is discarded.
    """

    assert (ID >= 0), 'impossible ID'

    #print(ID, "/", self.N)

    if ID >= self.N:
      # particle was found and lost by matlab between two updates
      return

    par = self.particles[ID]
    if (par.obs.shape[0] < 2):
      # particle seen only for one frame
      # not sufficient to add (s, a, r, s')
      self.particles.pop(ID)
      return

    last_val = 0.0
    if (not lost):
      last_val = self.critic(par.current).numpy()[0,0]


    # finish trajectory adding to memory the entire set of (obs, actions, logp, target)
    rews = np.append(par.rew, last_val)
    vals = np.append(par.val, last_val)

    # the next two lines compute Generalized Advantage Estimate
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv = np.append(self.adv, discount_cumsum(deltas, self.gamma * self.lam))

    # the next line computes rewards-to-go, to be targets for the value function
    self.target = np.append(self.target, discount_cumsum(rews, self.gamma)[:-1])

    # adds the trajectory (observables) to the memory
    self.obs = np.append(self.obs, par.obs, axis=0)

    # add the actions and the (log) probabilities to the memory
    # (if the particle is lost, last action/logp must be discarded)
    if (lost):
      self.actions = np.append(self.actions, par.act[:-1])
      self.logp = np.append(self.logp, par.logp[:-1])
    else:
      self.actions = np.append(self.actions, par.act)
      self.logp = np.append(self.logp, par.logp)

    # deletes the particle from memory
    self.particles.pop(ID) # CHECK IF THIS MESSES ORDER DURING TRAIN OR BETTER NOT DO HERE

  def normalize_adv(self):
    '''
    simple normalization trick of advantages for better convergence
    '''
    self.adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv)+0.1e-10)

  def train_step(self, epochs=10):
    '''
    train step, to be called after batch_size examples are taken,
    or the episode is done.
    In the first part, the (normalized) General Advantages Estimations are
    calculated.
    Then, it calculates the derivative for the clipped-PPO
    regarding the Actor, and applies them.
    In the end, updates the Critic to the target values.
    '''

    # FINISH ALL TRAJECTORIES
    for _ in self.particles:
      self.finish_path(lost=False, ID=0)  # CHECK THIS AS ABOVE!

    obs = self.obs
    opt = self.optimizer
    act = self.actions
    old_logp = self.logp
    # ----------------------------------
    self.normalize_adv()
    adv = self.adv


    for i in range(epochs):
      with tf.GradientTape() as tape:

        new_logp = tf.nn.log_softmax(self.policy(obs))
        new_logp_reduced = tf.reduce_sum(tf.one_hot(act, depth=self.n_actions) * new_logp, axis=1)  #CHECK PROPER DIMENSION
        logp_ratio = (new_logp_reduced - old_logp)
        ratio_reduced = tf.exp(logp_ratio)
        min_adv = np.where(adv>0, (1+self.CL)*adv, (1-self.CL)*adv)
        loss_ppo2 = -tf.reduce_mean(tf.minimum(ratio_reduced * adv, min_adv))
        # entropy term
        entropy_term = tf.reduce_mean(new_logp_reduced*tf.exp(new_logp_reduced))
        #
        loss = loss_ppo2 + self.en_coeff*entropy_term
        # ----

      grads = tape.gradient(loss, self.policy.trainable_variables)
      opt.apply_gradients(zip(grads, self.policy.trainable_variables))   #HOW MUCH???
      approx_kl = tf.reduce_mean(old_logp - new_logp_reduced)
      if (approx_kl > 1.5 * self.target_kl):
        print('iter: {}, approx_kl: {}'.format(i, approx_kl))
        break

    # -- CRITIC FITTING --------------------------
    self.critic.fit(x=obs, y=self.target, epochs=epochs, verbose=0)

    # --- reset internal values ----
    self.reset_batch()
    self.N = 0
