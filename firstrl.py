
import os
import numpy as np
import tensorflow as tf
import scipy.signal
import pickle

tf.keras.backend.set_floatx('float32')

# Documentation is optimized for pdoc:
# $ pip3 install pdoc
# $ pdoc firstrl.py


def discount_cumsum(x, discount):
    """
    Magic from rllab for computing discounted cumulative sums of vectors.

    Returns the discounted cummulative sum
    `[x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]`
    for an array-like input vector `x`.
    """
    ret = scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    if np.isnan(ret).any():
      print("NaNs in discount_cummsum")
      print(x)
      print(discount)

    ret[np.isnan(ret)] = 0
    return ret



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

  def __init__(self, observables, value):
    '''Starts the Trajectory with initial state `s0`.'''
    self.obs = [observables.reshape(-1)]
    self.val = [float(value)]
    self.act = []
    self.logp = []
    self.rew = []

  def add_action(self, action, logp):
    '''Adds an action `a` and the single probability for this action (not the whole distribution!).'''
    self.act.append(int(action))
    self.logp.append(float(logp))

  def add_state(self, reward, observables, value):
    '''Adds the reward `r` for a previous action and the new state `s'`.'''
    self.rew.append(float(reward))
    self.obs.append(observables.reshape(-1))
    self.val.append(float(value))


# ----------------------------------
# ACTUAL REINFORCEMENT LEARNING CODE
# ----------------------------------

class AgentActiveMatter():
  '''
  Complete class for Reinforcement Learning.
  Contains two Neural Networks for Actor and Critic, tracks and stores the trajectories
  '''


  def __init__(self, n_obs, lr_pi, lr_v, gamma, CL, en_coeff, lam, target_kl,
               n_actions, load_models, model_structure,
               train_actor=True, reinitialize_critic=False, actor_epochs=10,
               critic_epochs=1, episodic=False, bootstrap=False,
               **unused_parameters):
    '''
    Constructs a new RL Agent.

    If `load_models` is set, models are loaded from `load_models + '_critic/'` and
    `load_models + '_policy/'`, otherwise new models are constructed based on
    `n_obs`, `n_actions` and `model_structure`.

    `lr_pi, lr_v, gamma, CL, en_coeff, lam, target_kl` are training parameters.
    '''

    # internal knowledge
    self.optimizer = tf.optimizers.Adam(learning_rate=lr_pi)# optimizer
    self.gamma = gamma                                      # gamma for discount future rewards
    self.lam = lam                                          # lambda for GAE
    self.eps_clip = CL                                      # clipping parameter
    self.en_coeff = en_coeff                                # entropy coefficient
    self.target_kl = target_kl                              # target KL divergence for update early stop
    self.particles = []
    self.train_actor = train_actor                          # Whether or not the actor should be trained
    self.reinitialize_critic = reinitialize_critic          # Whether or not to reinitialize the critic
    self.critic_epochs = critic_epochs                      # How many epochs in training the critic
    self.actor_epochs = actor_epochs                        # How many epochs in training the actor
    self.episodic = episodic                                # Whether or not the task is episodic
    self.bootstrap = bootstrap                              # whether or not to bootstrap in episodic tasks

    # ------------------------------------------
    if (load_models):
      print('Loading from ' + load_models)

      self.critic = tf.keras.models.load_model(load_models + '_critic/', compile=False)
      self.critic.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_v), loss='mse')
      self.policy = tf.keras.models.load_model(load_models + '_policy/', compile=False)

      if self.reinitialize_critic:
        # Initialize Critic Neural Network new
        print('Reinitializing Critic')
        self.critic = tf.keras.Sequential(
          [
            # Input mask
            tf.keras.Input(shape=(n_obs,)),
            # Hidden Layers
            *[tf.keras.layers.Dense(size, activation=act) for size, act in model_structure],
            # Output Layer defining value of state
            tf.keras.layers.Dense(1, activation='linear')
          ]
        )

        # The critic layer is optimized with a default algorithm, so it can be compiled for speed
        self.critic.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_v), loss='mse')

      self.n_obs = self.critic.layers[0].input_shape[1]
      self.n_actions = self.policy.layers[-1].output_shape[1]
      self.reset_memory()

    else:
      print('Starting new model')
      assert model_structure, 'model structure is not defined!'

      self.n_obs = n_obs
      self.n_actions = n_actions
      self.reset_memory()

      # Actor Neural Network
      self.policy = tf.keras.Sequential(
        [
          # Input mask
          tf.keras.Input(shape=(self.n_obs,)),
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
          tf.keras.Input(shape=(self.n_obs,)),
          # Hidden Layers
          *[tf.keras.layers.Dense(size, activation=act) for size, act in model_structure],
          # Output Layer defining value of state
          tf.keras.layers.Dense(1, activation='linear')
        ]
      )

      # The critic layer is optimized with a default algorithm, so it can be compiled for speed
      self.critic.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_v), loss='mse')


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
    self.critic.save_weights(path + f'_critic/checkpoints/' + str(ckpt_id))
    self.policy.save_weights(path + f'_policy/checkpoints/' + str(ckpt_id))


  def reset_memory(self):
    '''Erases all cumulated experience'''
    self.observables = np.empty((0,self.n_obs))
    self.logp = np.empty((0))
    self.advantage = np.empty((0))
    self.estimated_return = np.empty((0))
    self.actions = np.empty((0), dtype=np.int8)


  def initialize(self, observables):
    '''Initialize new trajectories with `observables` of shape `(n_particles, n_obs)`.'''
    self.particles = [
      Trajectory(obs, self.critic(obs.reshape(1,-1))) for obs in observables
    ]


  def get_actions(self):
    '''
    Returns a list of actions `a` and their corresponding `log(P)` distribution from which
    they have been drawn, given the current state `s` of the Particles.
    '''

    # current observables of all particles
    observables = np.array([par.obs[-1] for par in self.particles])

    # action preference `h` is defined on interval (-Inf, Inf)
    preferences = self.policy(observables)

    logp = tf.nn.log_softmax(preferences).numpy()

    # draw random actions from the provided distributions
    try:
      actions = np.array([np.random.choice(self.n_actions, p=p) for p in np.exp(logp)])
    except ValueError:
      print('Nans in observables:')
      print(observables[np.argwhere(np.isnan(observables))])
      print('NaNs in logp:')
      print(logp[np.argwhere(np.isnan(logp))])

      logp[np.argwhere(np.isnan(logp))] = 0.25
      actions = np.array([np.random.choice(self.n_actions, p=p) for p in np.exp(logp)])

    actions[np.isnan(actions)] = 0

    # save for training
    for par, a, dist in zip(self.particles, actions, logp):
      par.add_action(a, dist[a])

    return actions, logp


  def add_environment_response(self, lost, observables, rewards, final=False):
    '''
    Update the Agent with the rewards for the last actions and the new states.

    If particles got lost since the last action, their IDs (= index position of
    the corresponding actions) need to be provided in order finish their trajectories.
    `observables` is expected to be of shape `(n_particle, n_obs)` wich may not
    contain the lost particles anymore, but may contain new particles at the end.
    '''

    # check inputs
    assert observables.shape[0] == rewards.shape[0], 'Inconsistent input of Obs and Rewards'

    # finish lost particles in reverse order to not mess up indices
    for ID_lost in sorted(lost, reverse=True):
      if ID_lost < len(self.particles):
        self.finish_trajectory(self.particles.pop(ID_lost))

    # Estimate value of new states
    if final and not self.bootstrap:
      values = np.zeros(rewards.shape)
    else:
      values = self.critic(observables).numpy().reshape(-1)

    # For debugging
    if np.isnan(rewards).any() or np.isnan(observables).any() or np.isnan(values).any():
      print("NaNs in add_environment_response")
      print(observables)
      print(rewards)
      print(values)

    # This is a bad fix, but no I don't know the real problem yet (15.02.22)
    observables[np.isnan(observables)] = 0
    rewards[np.isnan(rewards)] = 0
    values[np.isnan(values)] = 0

    # Update particles list
    for i, (rew, obs, val) in enumerate(zip(rewards, observables, values)):
      if i < len(self.particles):
        self.particles[i].add_state(rew, obs, val)
      else:
        self.particles.append(Trajectory(obs, val))

    return values


  def finish_trajectory(self, traj):
    '''
    Consumes a trajectory and converts it into experience that can be used to
    train the models (i.e. calculate the Genaral Advantage Estimator and
    bootstrapped Return / Rewards-to-go).
    If the particle got lost, the last action is discarded.

    **FROM SPINNING UP's PPO CODE:**
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
    '''

    if len(traj.obs) < 2:
      # particle seen only for one frame
      # not sufficient to add (s, a, r, s')
      return

    if len(traj.act) > len(traj.rew):
      # no reward for the last action, this happens when the particle got lost,
      # so the last action should be discarded
      traj.act.pop()
      traj.logp.pop()

    # just double check ...
    assert len(traj.obs) - 1 == len(traj.val) - 1 == len(traj.act) == len(traj.logp) == len(traj.rew)
    rews = np.array(traj.rew)
    vals = np.array(traj.val)

    # adds the states (except the last) and actions choosen for those states to the memory
    self.observables = np.append(self.observables, np.array(traj.obs[:-1]), axis=0)
    self.actions = np.append(self.actions, traj.act)
    self.logp = np.append(self.logp, traj.logp)

    # compute Generalized Advantage Estimate to train policy
    # (for details, see https://arxiv.org/abs/1506.02438v6)
    deltas = rews + self.gamma * vals[1:] - vals[:-1]

    self.advantage = np.append(self.advantage, discount_cumsum(deltas, self.gamma * self.lam))

    # compute estimated return (+ bootstrap value) as target for the value function
    self.estimated_return = np.append(self.estimated_return, discount_cumsum(np.append(rews, vals[-1]), self.gamma)[:-1])


  def finish_episode(self):
    '''Finishes all trajectories.'''
    while self.particles:
      self.finish_trajectory(self.particles.pop())


  def train_step(self):
    '''
    Train the model with accumulated experience.

    In the first part, the (normalized) General Advantages Estimations are
    calculated.
    Then, it calculates the derivative for the clipped-PPO
    regarding the Actor, and applies them.
    In the end, updates the Critic to the target values.
    '''

    # convert all recorded trajectories in experience
    self.finish_episode()

    # normalize advantage for better convergence
    adv_std = np.std(self.advantage)

    if (adv_std > 0.1e-1):
        self.advantage = (self.advantage - np.mean(self.advantage)) / adv_std

    # For debugging
    if np.isnan(self.advantage).any():
      print("NaNs in train_step in advantages")
      print(self.advantage)
      self.advantage[np.isnan(self.advantage)] = 0


    # -- POLICY FITTING --
    if self.train_actor:
      for i in range(self.actor_epochs):
        # TensorFlow GradientTape magic to do numeric derivatives
        # (all operations in the `with` block are recorded somehow in the C++ backend)
        with tf.GradientTape() as tape:

          # calculate loss function with clipped PPO:
          #
          # for details, see
          # - https://arxiv.org/abs/1707.06347
          # - https://spinningup.openai.com/en/latest/algorithms/ppo.html
          #
          # here:
          # π_θ_k(a|s) == exp(self.logp)
          # π_θ(a|s) == exp(new_logp)
          # A^{π_θ_k}(s,a) == self.adv
          # ε == self.eps_clip
          new_logp_dist = tf.nn.log_softmax(self.policy(self.observables))
          new_logp = tf.reduce_sum(tf.one_hot(self.actions, depth=self.n_actions) * (new_logp_dist), axis=1)
          probability_ratio = tf.exp(new_logp - self.logp)
          max_adv = np.where(self.advantage > 0, (1+self.eps_clip) * self.advantage, (1-self.eps_clip) * self.advantage)
          loss_ppo2 = -tf.reduce_mean(tf.minimum(probability_ratio * self.advantage, max_adv))

          # additional loss function to keep finite entropy:
          # DKL(P_uniform || P_new) = -log(P_new) calculates the Kullback-Leibler divergence
          # between the current distribution and a uniform distribution. If en_coeff > 0 this
          # term biases the loss function towards more entropy, to keep a minimum amount of
          # explorative behavior in the policy
          loss = loss_ppo2 - self.en_coeff * tf.reduce_mean(new_logp_dist)

        # calculate numerical derivative of the loss function in respect to the policy parameters θ
        grads = tape.gradient(loss, self.policy.trainable_variables)

        # optimize the policy along these gradients
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        # early stopping to prevent overfitting
        # (should already be accomplished by the PPO algorithm (?))
        approx_kl = tf.reduce_mean(self.logp - new_logp)
        if (approx_kl > 1.5 * self.target_kl):
          print('Stopping policy optimication after epoch {}, approx_kl: {}'.format(i, approx_kl))

          # in this case, there is something wrong with the training data and I
          # want to know what, so self.observables, self.advantage and
          # self.estimated_return are saved for examination
          print('Dumping observables, advantages and estimated returns')
          if not os.path.isdir("./nan_dump"): os.mkdir("./nan_dump")

          with open("./nan_dump/observables.pickle", "wb") as obs_file:
            pickle.dump(self.observables, obs_file, protocol=pickle.HIGHEST_PROTOCOL)
          with open("./nan_dump/observables.pickle", "wb") as obs_file:
            pickle.dump(self.observables, obs_file, protocol=pickle.HIGHEST_PROTOCOL)
          with open("./nan_dump/advantage.pickle", "wb") as adv_file:
            pickle.dump(self.advantage, adv_file, protocol=pickle.HIGHEST_PROTOCOL)
          with open("./nan_dump/estimated_return.pickle", "wb") as est_ret_file:
            pickle.dump(self.estimated_return, est_ret_file, protocol=pickle.HIGHEST_PROTOCOL)
          break

    # -- CRITIC FITTING --
    if np.isnan(self.observables).any(): # ZZZ
      print("NaNs in self.observables in critic fitting")
      print(self.observables)
      self.observables[np.isnan(self.observables)] = 0
    if np.isnan(self.estimated_return).any():
      print("NaNs in self.estimated_return in critic fitting")
      print(self.estimated_return)
      self.estimated_return[np.isnan(self.estimated_return)] = 0
    self.critic.fit(x=self.observables, y=self.estimated_return, epochs=self.critic_epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)], verbose=0)

    # clean up
    self.reset_memory()
