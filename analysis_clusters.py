#!/usr/bin/env python
# TEST WITH MD

import numpy as np
import sys
import scipy
#import tensorflow as tf

from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

# FOR KL convergence
def Pi(obs, policy):
    logp = policy(np.array(obs))
    prob = np.exp(logp)
    prob = prob / np.sum(prob,axis=1,keepdims=True)
    return prob

def from_policy_to_actions(Pi):
    '''
    takes distribution of log probabilities over discrete set of actions
    and gives out one randomly, after normalization
    MUST RETURN only one value: index of action!
    '''
    action=np.random.choice(4,p=Pi)
    return action

def KL_symm(A,B):
    return 0.5*np.mean(entropy(A,B,axis=1)+entropy(B,A,axis=1))

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def map_local_swirl(pos, orient, sigma, rotation):
    com = np.mean(pos,axis=0)
    versor = (pos-com) / np.linalg.norm(pos - com)
    p_swirl = orient[:,0]*versor[:,1]-orient[:,1]*versor[:,0]
    local_swirl = np.zeros((pos.shape[0],8))
    local_swirl[:,:2] = pos
    local_swirl[:,2:4] = orient
    for i, par in enumerate(pos):
        dist = np.linalg.norm(pos-par, axis=1)
        local_swirl[i,4] = np.sum(gaussian(dist, 0, sigma))
        local_swirl[i,5] = np.dot(gaussian(dist, 0, sigma), p_swirl)
        local_swirl[i,6] = rotation[i]
        local_swirl[i,7] = np.dot(gaussian(dist, 0, sigma), rotation)
    return local_swirl
    
def how_much_rot(traj, last, first):
    theta_new = np.arctan2(particles[last, :, 4], particles[last, :, 3])
    theta_old = np.arctan2(particles[first, :, 4], particles[first, :, 3])
    rotation = theta_new - theta_old
    rotation = rotation - np.floor(rotation / 2 /np.pi + 0.5)*2*np.pi
    return rotation

if __name__ == "__main__":
    # READS FOOD_REW AS INPUT    
    traj_name = sys.argv[1]
    Np = np.int(sys.argv[2])
    rot_avg = np.int(sys.argv[3])

    momo = np.loadtxt(traj_name)
    Nframes = momo.shape[0] // (Np+1)
    frames = np.array(np.split(momo, Nframes))
    particles = frames[:,1:,1:]
    food = frames[:,1,1:]

    for iframe in range(Nframes):
        hull = ConvexHull(particles[iframe,:,:2])
        area, volume = hull.area, hull.volume
        momo = PCA(n_components=2).fit(particles[iframe,:,:2])
        a = momo.explained_variance_ratio_[1]
        b = momo.explained_variance_ratio_[0]
        elongation = np.sqrt(b - a) / np.sqrt(b)
        if (iframe >= rot_avg):
            rotation = how_much_rot(particles, iframe, iframe-rot_avg)
            local_swirl = map_local_swirl(particles[iframe,:,:2], particles[iframe,:,3:5], 15, rotation)
            np.savetxt('local_swirl_'+traj_name+'_frame{:04d}.xyz'.format(iframe), local_swirl)
        print(iframe, volume, elongation)
        