# Analysis
import numpy as np
import tensorflow as tf
from scipy.stats import entropy as sk_entropy
import evolve_fortran as evolve

# Loading models - checking consistency

# Change here name of model-------
models_rootname = 'models_demix'
# --------------------------------

print('Loading from ' + models_rootname)

critic = tf.keras.models.load_model(models_rootname+'_critic/')
policy = tf.keras.models.load_model(models_rootname+'_policy/')

loaded_input_dim = critic.layers[0].input_shape[1]
loaded_output_dim = policy.layers[-1].output_shape[1]


# Naming of internal layers, for future analysis
dense_0_policy = policy.get_layer('dense').output
dense_1_policy = policy.get_layer('dense_1').output

policy_first_layer=tf.keras.models.Model(inputs=policy.input, outputs=dense_0_policy)
policy_second_layer=tf.keras.models.Model(inputs=policy.input, outputs=dense_1_policy)
# -----------------------------------------------

def Pi_PiEntropy(obs):
    logp = policy(np.array([obs]))
    prob = np.exp(logp)
    prob = prob / np.sum(prob)
    return prob, sk_entropy(prob[0])

# Create Fake Scenario.

# 30 particles around
# 1 particle is in the center
# 1 particle is "just behind it": needed to have even numbers, it is not seen



# Scenario 0: half / half
N = 32
p0 = np.zeros((N,4))
p0[16:,3]=1
p0[:16,3]=0
angles = np.linspace(-np.pi/2., 3/2.*np.pi, 30, endpoint=False)
for ia,a in enumerate(angles):
    p0[ia+1,:3] = 20*np.array([np.cos(a), np.sin(a), 0])

# Scenario 1: 90 / 90 / 180
N = 18
p1 = np.zeros((N,4))
p1[9:,3]=1
p1[:9,3]=0
angles = np.linspace(0, np.pi, 16, endpoint=False)
for ia,a in enumerate(angles):
    p1[ia+1,:3] = 20*np.array([np.cos(a), np.sin(a), 0])

# Scenario 2: mixed half
N = 32
p2 = np.zeros((N,4))
p2[16:,3]=1
p2[:16,3]=0
angles = np.linspace(-np.pi/3., 2./3.*np.pi, 15, endpoint=False)
for ia,a in enumerate(angles):
    p2[ia+1,:3] = 20*np.array([np.cos(a), np.sin(a), 0])
angles = np.linspace(np.pi/3., 4./3.*np.pi, 15, endpoint=False)
for ia,a in enumerate(angles):
    p2[ia+16,:3] = 20*np.array([np.cos(a), np.sin(a), 0])



# From Fake Scenario to observables
# Particle in the middle rotates
angles = np.linspace(-np.pi/2., 3/2.*np.pi, 100)
for ip, p in enumerate([p0, p1, p2]):
    N = p.shape[0]
    result = open('results_scenario{}.dat'.format(ip), "w")
    np.savetxt('scenario{}.dat'.format(ip), p)
    for a in angles:
        p[-1,2] = a
        p[0,:3] = [-np.cos(a),-np.sin(a), 0] # shadow particle to 
        obs, rewards = evolve.get_o_r_mix_tasks(
            p[:,0], p[:,1], p[:,2],   # X , Y, Theta    respectively
            1.0, 2, -1,               # mixing cost, demixing mode, switch_flag
            10, N)                    # N_Obs, N
        # Only last particle actions are of interest
        ob = obs[-1]
        pi, pientropy = Pi_PiEntropy(ob)
        result.write('{} {} {} {} {} {} {}\n'.format(a, np.argmax(pi[0]), pi[0,0], pi[0,1], pi[0,2], pi[0,3], pientropy/np.log(4.)))