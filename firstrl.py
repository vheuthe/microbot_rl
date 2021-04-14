
import numpy as np
import tensorflow as tf
import scipy.signal

tf.keras.backend.set_floatx('float32')

# Documentation is optimized for pdoc:
# $ pip3 install pdoc
# $ pdoc firstrl.py

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
    Magic from rllab for computing discounted cumulative sums of vectors.

    Returns the discounted cummulative sum 
    `[x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]` 
    for an array-like input vector `x`.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# This is an ausiliary class that memorize the trajectory of a single particle

class Trajectory():
  '''
  Utility class to store a single trajectory

  A trajectory is defined as a sequence of states, actions and rewards: 
  `s -> a -> r,s' -> a' -> r',s''`.
  Each state is stored in the form of the corresponding set of observables
  and the current value estimate of that state. Alongside with each action,
  also the probability distribution from which the action was drawn is stored
  (it is needed later to calculate the loss function).

  This class does not check correct order of adding states and actions, the
  caller has to take care about proper usage.
  '''

  def __init__(self, observables):
    '''Starts the Trajectory with initial state `s0`.'''

    # check obs shape is (1, n_inputs)!
    self.current = observables
    self.obs = np.empty((0,observables.shape[1]))
    self.act = np.empty((0), dtype=np.int8)
    self.rew = []
    self.val = []
    self.logp = []

  def add_action(self, action, logp):
    '''Adds an action `a` and the corresponding probability distribution.'''
    self.act = np.append(self.act, action)
    self.logp = np.append(self.logp, logp)

  def add_state(self, reward, observables, value):
    '''Adds the reward `r` for a previous action and the new state `s'`.'''
    self.obs = np.append(self.obs, self.current, axis=0)
    self.current = observables
    self.val = np.append(self.val, value)
    self.rew = np.append(self.rew, reward)

# ----------------------------------
# ACTUAL REINFORCEMENT LEARNING CODE
# ----------------------------------

class AgentActiveMatter():
  '''
  Complete class for Reinforcement Learning.
  Contains two Neural Networks for Actor and Critic, tracks and stores the trajectories
  '''
  def __init__(self, input_dim, output_dim, lrPI, lrV, gamma, CL, en_coeff, lam, target_kl,
               restart_models = False, models_rootname='./model', model_structure=[(32, 'relu'),(16, 'relu'),(16, 'relu')],
               **unused_parameters):

    # internal knowledge
    self.optimizer = tf.optimizers.Adam(learning_rate=lrPI) # optimizer
    self.gamma = gamma                                      # gamma for discount future rewards
    self.lam = lam                                          # lambda for GAE
    self.CL = CL                                            # clipping parameter
    self.en_coeff = en_coeff                                # entropy coefficient
    self.lrPI = lrPI                                        # learning rate policy
    self.lrV = lrV                                          # learning rate value
    self.target_kl = target_kl                              # target KL divergence for update early stop

    self.particles = []                         # initialize memory (to zero)

    # ------------------------------------------
    if (restart_models):
      print('Loading from ' + models_rootname)

      self.critic = tf.keras.models.load_model(models_rootname+'_critic/')
      self.policy = tf.keras.models.load_model(models_rootname+'_policy/')

      loaded_input_dim = self.critic.layers[0].input_shape[1]
      loaded_output_dim = self.policy.layers[-1].output_shape[1]

      self.input_dim = loaded_input_dim
      self.n_actions = loaded_output_dim
      self.reset_memory()    

    else:
      print('Starting new model')
      
      self.input_dim = input_dim
      self.n_actions = output_dim
      self.reset_memory()    

      assert model_structure, 'model structure is not defined!'

      # Actor Neural Network
      self.policy = tf.keras.Sequential(
        [
          # Input mask
          tf.keras.Input(shape=(self.input_dim,)),
          # Hidden Layers
          *[tf.keras.layers.Dense(size, activation=act) for size, act in model_structure],
          # Output Layer defining actions
          tf.keras.layers.Dense(self.n_actions, activation='linear')
        ]
      )

      # Critic Neural Network
      self.critic = tf.keras.Sequential(
        [
          # Input mask
          tf.keras.Input(shape=(self.input_dim,)),
          # Hidden Layers
          *[tf.keras.layers.Dense(size, activation=act) for size, act in model_structure],
          # Output Layer defining value of state
          tf.keras.layers.Dense(1, activation='linear')
        ]
      )

      # The critic layer is optimized with a default algorithm, so it can be compiled for speed
      self.critic.compile(optimizer=tf.optimizers.Adam(learning_rate=lrV), loss='mse')



#  -----------------------------
  def save_models(self, path):
    '''
    Saves critic and policy models in TensorFlow format at `path + '_critic/'` and `path + '_policy/'`.
    '''
    self.critic.save(path + '_critic')
    self.policy.save(path + '_policy')

  def save_weights(self, path, ckpt_id):
    '''
    Saves the weights of critic and policy in TensorFlow checkpoint-format,
    the structure of the model has to be saved separately.
    '''
    self.critic.save_weights(path + '_critic/checkpoints/' + str(ckpt_id))
    self.policy.save_weights(path + '_policy/checkpoints/' + str(ckpt_id))


  def reset_memory(self):
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
    self.particles = [Trajectory(o.reshape(1,self.input_dim)) for o in obs]    # creates list of SAM objects, where to store individual particles

  def add_env_timeframe(self, lost, new_obs, rewards, isdone=False):
    '''
    receives information about the present time step
    from the outside world:
    - lost is list of IDs of particles that have been lost in tracking !! AS NUMBERED IN THE PREVIOUS FRAME!!
    - new_obs is a numpy array of observables of all particles (N_particle, n_input_dim)
    '''

    # pop lost particles in reverse order to not mess up indices
    for ID_lost in sorted(lost, reverse=True):
      if ID_lost < len(self.particles):
        self.finish_path(self.particles.pop(ID_lost), True)

    vals = []
    assert new_obs.shape[0] == rewards.shape[0], 'Inconsistent input of Obs and Rewards'

    for i, (obs, rew) in enumerate(zip(new_obs, rewards)):
      obs = obs.reshape(1,-1)
      if i < len(self.particles):
        val = self.critic(self.particles[i].current).numpy()[0,0]
        self.particles[i].add_state(rew, obs, val)
        vals.append(val)
      else:
        self.particles.append(Trajectory(obs))
        vals.append(np.nan)

    # Ends current episode
    if isdone:
      while self.particles:
        self.finish_path(self.particles.pop())

    return vals

  def finish_episode(self):
    while self.particles:
      self.finish_path(self.particles.pop())

  def get_actions(self, flag_logp=False):
    '''
    returns numpy array of actions
    to take given the current observables
    and updates the particles with action, current value and logp
    '''
    actions = []
    pi_logp_all = np.empty((0,self.n_actions))

    for par in self.particles:
      # for each particle,
      # its current observable is used to choose an action
      o = par.current
      pi_logits = self.policy(o)
      pi_logp = tf.nn.log_softmax(pi_logits)
      a = from_policy_to_actions(pi_logp)
      logp = pi_logp[0, a]
      par.add_action(a, logp)
      #print('HERE #par.actions: {}'.format(len(par.act)))
      #print('HERE #par.rewards: {}'.format(len(par.rew)))
      actions = np.append(actions, a)
      pi_logp_all = np.append(pi_logp_all, pi_logp, axis=0)
    if (flag_logp):
        return actions.astype(int), pi_logp_all
    else:
        return actions.astype(int)
    

  def finish_path(self, particle, lost = False):
    """
    **FROM SPINNING UP's PPO CODE**
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
    
    In this realization, each particle is considered as an individual
    trajectory.
    If the particle is lost, it is considered as it received a "done"
    signal, and the last action is discarded.
    """

    if (particle.obs.shape[0] < 2):
      # particle seen only for one frame
      # not sufficient to add (s, a, r, s')
      return

    # our particles are imortale, so we always have infinite horizon
    last_val = self.critic(particle.current).numpy()[0,0]

    # finish trajectory adding to memory the entire set of (obs, actions, logp, target)
    rews = np.append(particle.rew, last_val)
    vals = np.append(particle.val, last_val)

    # the next two lines compute Generalized Advantage Estimate
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv = np.append(self.adv, discount_cumsum(deltas, self.gamma * self.lam))

    # the next line computes rewards-to-go, to be targets for the value function
    self.target = np.append(self.target, discount_cumsum(rews, self.gamma)[:-1])

    # adds the trajectory (observables) to the memory
    self.obs = np.append(self.obs, particle.obs, axis=0)

    # add the actions and the (log) probabilities to the memory
    # (if the particle is lost, last action/logp must be discarded)
    if (lost):
      self.actions = np.append(self.actions, particle.act[:-1])
      self.logp = np.append(self.logp, particle.logp[:-1])
    else:
      self.actions = np.append(self.actions, particle.act)
      self.logp = np.append(self.logp, particle.logp)


  def normalize_adv(self):
    '''
    simple normalization trick of advantages for better convergence
    '''
    adv_std = np.std(self.adv)
    if (adv_std > 0.1e-1):
        self.adv = (self.adv - np.mean(self.adv)) / adv_std


  def train_step(self, epochs=10):
    '''
    Train the model with accumulated experience.

    In the first part, the (normalized) General Advantages Estimations are
    calculated.
    Then, it calculates the derivative for the clipped-PPO
    regarding the Actor, and applies them.
    In the end, updates the Critic to the target values.
    '''

    # FINISH ALL TRAJECTORIES
    while self.particles:
      self.finish_path(self.particles.pop()) 

    obs = self.obs
    opt = self.optimizer
    act = self.actions
    old_logp = self.logp
    # ----------------------------------
    
    self.normalize_adv()
    adv = self.adv
    
    
    for _i in range(epochs):
      with tf.GradientTape() as tape:

        new_logp = tf.nn.log_softmax(self.policy(obs))
        new_logp_reduced = tf.reduce_sum(tf.one_hot(act, depth=self.n_actions) * new_logp, axis=1)
        logp_ratio = (new_logp_reduced - old_logp)
        ratio_reduced = tf.exp(logp_ratio)
        min_adv = np.where(adv>0, (1+self.CL)*adv, (1-self.CL)*adv)
        loss_ppo2 = -tf.reduce_mean(tf.minimum(ratio_reduced * adv, min_adv))
        # approx KL divergence respect to flat - to reduce certainty.
        # DKL(P_flat || Pi_new)
        DKL_term = tf.reduce_mean(-new_logp)
        # 
        loss = loss_ppo2 + self.en_coeff*DKL_term
        # ----

      grads = tape.gradient(loss, self.policy.trainable_variables)
      opt.apply_gradients(zip(grads, self.policy.trainable_variables))   #defined in self.optimizer
      approx_kl = tf.reduce_mean(old_logp - new_logp_reduced)
      if (approx_kl > 1.5 * self.target_kl):
        #print('iter: {}, approx_kl: {}'.format(i, approx_kl))
        break

    # -- CRITIC FITTING AND LOGGING ----------------------
    #self.critic.fit(x=obs, y=self.target, epochs=epochs*20, verbose=0)
    self.critic.fit(x=obs, y=self.target, epochs=epochs*20, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)], verbose=0)
    #loss_history = np.array(history_callback.history["loss"])
    #with open("loss_history.txt", "a") as f:
    #    np.savetxt(f, loss_history)

    # --- reset internal values ----
    self.reset_memory()
