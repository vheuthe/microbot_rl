The code in microbot_rl was written by Emanuele Panizon, Robert Löffler and Veit-Lorenz Heuthe

# microbot_rl
microbot_rl contains Multi-Agent Reinforcenemt Learning (MARL) and simulation code for training microrobot swarms to rotate or transport a large rod.
Each microbot is a seperate RL agent and acts on the basis of local information.
Training is performed according to the centralized-training decentralized-execution (CTDE) paradigm, making the system fully decentralized in deployment.

## MARL algorithm
The code for the MARL algorithm is found in *firstrl.py*. It contains an *AgentActiveMatter* class that stores trajectories in the form of observables (representations of the state for each robot), actions and rewards and contains methods for training the actor and critic networks.
The MARL based on an actor-critic approache that is optimized using PPO-clip. Details about the PPO algorithm can be found on [arxiv](https://arxiv.org/abs/1707.06347) and [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

## Simulation engine
*fortran/evolve_environment.f95* contains the simulation engine.
It has two subroutines: *evolve_md_rod* evolves the environment in time and *get_o_r_rod* computes the observables and rewards for each robot (for details about the implemented rewarding schemes see the [Rewarding schemes](#rewarding-schemes) section).
Using the *Makefile*, the fortran code can be compiled to be imported in python.

## Environment
The details of the environment are dealt with by *environments/rod.py*.
It contains a *MD_ROD* class that contains the robots, the rod and a potential target (for targeted transport) and potential obstacles and calles the [fortran subroutines](#simulation-engine) to evolve and compute the state representations (observables).
The environment also computed rewards for more advanced [rewarding schemes] than the fortran code does.

## Rewarding schemes
There are three rewarding schemes to choose from:
- Torque-based rewards, where each robot receives a reward based on the torque it exerts on the rod. This naturally only works for the task of rotating the rod.
- Team rewards, where each robot receives the same reward, which is proportional to the performance of the whole swarm.
- Counterfactual rewards. In this rewarding scheme, the lase timestep is repeated for each robot and the considered robot is removed from this re-simulation step. Then, the contribution of the considered robot to the swarm performance is estimated based on the difference in performance that the presence of this robot made.
Rewards are computed by *environments/rod.py* or the fortran engine.

## Training procedure
*learning_rod.py* contains the training procedure. It contains default parameters, sets up the [environment](#environment) and the [RL-agent](#marl-algorithm).
The *do_array_task* or *do_task* functions perform training and save the results in .h5 format.
*do_array_task* can load a .json file containing a dictionary of parameters and then performs training for each parameter combination. To train multiple models with each parameter combination include a 'rep' parameter that has a list of number (like `rep:[1, 2, 3, 4, 5]`)

## Parameters
There are many parameters to tune the microbot training:
### General
`N`: Number of microbots in the swarm.

`mode`: Specifies the task the microbots are supposed to solve: 3 corresponds to rod rotation, 4 to rotation in clockwise direction (only supports torque-rewards) and 7 to targeted transport.

### Rewarding scheme
`CR_mode`: 'non_ex' (default) or 'passive' for making the consiered robot either non-existent of passive during resimulation-steps.

`CR_prefact`: Prefactor for the counterfactual rewards.

`CR_noise`: Whether or not to include noise in the re-simulation steps. 'mixed' is the default which does not include noise in the re-simulatin steps to reduce the variance in the reward signal.

`parallelize_CR`: Flag for parallelizing the counterfactual rewards (using the multiprocessing module in python).

`n_processes`: Number of parallel processes when `parallelize_CR`=`true`.

`rew_cutoff`: Cutoff distance to rod for rewarding in `rew_mode` 'team' or 'CR'.

`rew_mode`: Specifies the rewarding scheme, can be 'torque', 'team' or 'CR' for torque-based rewards (only work for rotation task, mode=3), team based rewards or counterfactual rewards.

`team_reward_mode`: 'team', 'close' or 'touch' determines if all robots ('team'), the ones closer than 'rew_cutoff' to the rod ('close') or only the robots that are touching the rod ('touch') are rewarded in `rew_mode` 'team'.

### MARL algorithm
`actor_epochs`: Number of epochs in one actor update.

´critic_epochs`: Number of epochs in one critic update.

`load_models`: `None` to initialize models (actor and critic) randomly or specify path to load models from.

`model_structure`: Structure of the neural networks (for both actor and critic). Only specifies hidden layers, since input and output node numbers are determined by the number of actions and the number of observables.

`CL`, `gamma`, `lam`, `lr_pi`, `lr_v`, `target_kl`:
Parameters for tuning the optimization of the actor and critic networks.

### Training procedure
`episodic`: Flag for doing truly episodic training (meaning the task can be achieved, which discontinues the episode).

`episodic_eval`: Whether the evaluation should be done truly episodic.

`train_frames`, `eval_frames`: Number of frames in each training/evaluation episode.

`n_train_ep`, `n_eval_ep`: Number of training/evaluation episodes.

`train_pause`: Number of frames before the models are updated.

`train_actor`, `train_critic`: Flags for training actor and critic.

`record_traj`: Whether to recort the positions of each microbot during the whole training.

`eval_only`: Flag for only running evaluation and no training.

### Observables
`obs_type`: '1overR' or '1overR2' for the sensing of the microbots to decay with 1/R or 1/R^2 (where R is the distance to the sensed object).

`cones`: Number of "vision" cones for each robot.

`cone_angle`: The total perception angle (always to the front).

### Physics of the environment
`fr_rod`: Friction coefficient of the rod. Determines, how much force is necessary to move/rotate the rod with  a certain velocity.

`mu_k`: Coefficient for friction between the robots and the rod. `mu_k`=0: no friction, `mu_k`>1: friction.

`vel_act`, `vel_tor`, `vel_noise_fact`, `rot_noise_fact`, `torque`, `Dt`, `Dr`: Parameters for tuning the motion of the microrobots.

### Obstacles
`obst_conf`: Obstacle configuration, either 'random' for a pseudo-random configuration or 'wall' for a wall-shaped obstacle.

´obst_vision`: Flag that decides whether or not the microbots can sense the obstacles.

`use_obst`: Flag for including obstacles in the environment.