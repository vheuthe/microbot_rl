
import numpy as np
import tensorflow as tf
import scipy.signal

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
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



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


  def __init__(self, input_dim, lrPI, lrV, gamma, CL, en_coeff, lam, target_kl,
               nActions=4, load_models = None, model_structure=[(32, 'relu'),(16, 'relu'),(16, 'relu')],
               **unused_parameters):
    '''
    Constructs a new RL Agent.

    If `load_models` is set, models are loaded from `load_models + '_critic/'` and
    `load_models + '_policy/'`, otherwise new models are constructed based on
    `input_dim`, `output_dim` and `model_structure`.

    `lrPI, lrV, gamma, CL, en_coeff, lam, target_kl` are training parameters.
    '''

    # internal knowledge
    self.optimizer = tf.optimizers.Adam(learning_rate=lrPI) # optimizer
    self.gamma = gamma                                      # gamma for discount future rewards
    self.lam = lam                                          # lambda for GAE
    self.CL = CL                                            # clipping parameter
    self.en_coeff = en_coeff                                # entropy coefficient
    self.target_kl = target_kl                              # target KL divergence for update early stop
    self.particles = []

    # ------------------------------------------
    if (load_models):
      print('Loading from ' + load_models)

      self.critic = tf.keras.models.load_model(load_models + '_critic/')
      self.policy = tf.keras.models.load_model(load_models + '_policy/')

      self.input_dim = self.critic.layers[0].input_shape[1]
      self.nActions = self.policy.layers[-1].output_shape[1]
      self.reset_memory()

    else:
      print('Starting new model')
      assert model_structure, 'model structure is not defined!'

      self.input_dim = input_dim
      self.nActions = nActions
      self.reset_memory()

      # Actor Neural Network
      self.policy = tf.keras.Sequential(
        [
          # Input mask
          tf.keras.Input(shape=(self.input_dim,)),
          # Hidden Layers
          *[tf.keras.layers.Dense(size, activation=act) for size, act in model_structure],
          # Output Layer defining actions
          tf.keras.layers.Dense(self.nActions, activation='linear')
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
    '''Erases all cumulated experience'''
    self.observables = np.empty((0,self.input_dim))
    self.logp = np.empty((0))
    self.advantage = np.empty((0))
    self.estimated_return = np.empty((0))
    self.actions = np.empty((0), dtype=np.int8)


  def initialize(self, observables):
    '''Initialize new trajectories with `observables` of shape `(n_particles, input_dim)`.'''
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

    # Some NaNs appeared here...
    logp[np.isnan(logp)] = 0
    logp = logp/sum(logp)

    # draw random actions from the provided distributions
    actions = [np.random.choice(self.nActions, p=p) for p in np.exp(logp)]

    # save for training
    for par, a, dist in zip(self.particles, actions, logp):
      par.add_action(a, dist[a])

    return actions, logp


  def add_environment_response(self, lost, observables, rewards):
    '''
    Update the Agent with the rewards for the last actions and the new states.

    If particles got lost since the last action, their IDs (= index position of
    the corresponding actions) need to be provided in order finish their trajectories.
    `observables` is expected to be of shape `(n_particle, input_dim)` wich may not
    contain the lost particles anymore, but may contain new particles at the end.
    '''

    # check inputs
    assert observables.shape[0] == rewards.shape[0], 'Inconsistent input of Obs and Rewards'

    # finish lost particles in reverse order to not mess up indices
    for ID_lost in sorted(lost, reverse=True):
      if ID_lost < len(self.particles):
        self.finish_trajectory(self.particles.pop(ID_lost))

    # Estimate value of new states
    values = self.critic(observables).numpy().reshape(-1)

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

    # adds the states (except the last) and actions choosen for those states to the memory
    self.observables = np.append(self.observables, np.array(traj.obs[:-1]), axis=0)
    self.actions = np.append(self.actions, traj.act)
    self.logp = np.append(self.logp, traj.logp)

    # as our episodes have infinite horizon, add the value of the last state as last reward
    # to bootstrap the estimate of the return. The true return can only be calculated if the
    # episode reaches a final state, which is not possible in this scenario.
    # (this additional reward is ignored in the calculation of GAE.)
    rews = np.append(traj.rew, traj.val[-1])
    vals = np.array(traj.val)

    # compute Generalized Advantage Estimate to train policy
    # (for details, see https://arxiv.org/abs/1506.02438v6)
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.advantage = np.append(self.advantage, discount_cumsum(deltas, self.gamma * self.lam))

    # compute estimated return as target for the value function
    self.estimated_return = np.append(self.estimated_return, discount_cumsum(rews, self.gamma)[:-1])


  def finish_episode(self):
    '''Finishes all trajectories.'''
    while self.particles:
      self.finish_trajectory(self.particles.pop())


  def train_step(self, epochs=10):
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

    # -- POLICY FITTING --
    for i in range(epochs):
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
        # ε == self.CL
        new_logp_dist = tf.nn.log_softmax(self.policy(self.observables))
        new_logp = tf.reduce_sum(tf.one_hot(self.actions, depth=self.nActions) * new_logp_dist, axis=1)
        probability_ratio = tf.exp(new_logp - self.logp)
        max_adv = np.where(self.advantage > 0, (1+self.CL) * self.advantage, (1-self.CL) * self.advantage)
        loss_ppo2 = -tf.reduce_mean(tf.minimum(probability_ratio * self.advantage, max_adv))

        # additional loss function to keep finite entropy:
        # DKL(P_uniform || P_new) = -log(P_new) calculates the Kullback-Leibler divergence
        # between the current distribution and a uniform distribution. If en_coeff > 0 this
        # term biases the loss function towards more entropy, to keep a minimum amount of
        # explorative behavior in the policy
        loss = loss_ppo2 + self.en_coeff * tf.reduce_mean(-new_logp_dist)

      # calculate numerical derivative of the loss function in respect to the policy parameters θ
      grads = tape.gradient(loss, self.policy.trainable_variables)
      # optimize the policy along these gradients
      self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

      # early stopping to prevent overfitting
      # (should already be accomplished by the PPO algorithm (?))
      approx_kl = tf.reduce_mean(self.logp - new_logp)
      if (approx_kl > 1.5 * self.target_kl):
        print('Stopping policy optimication after epoch {}, approx_kl: {}'.format(i, approx_kl))
        break

    # -- CRITIC FITTING --
    self.critic.fit(x=self.observables, y=self.estimated_return, epochs=epochs*20, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)], verbose=0)

    # clean up
    self.reset_memory()
