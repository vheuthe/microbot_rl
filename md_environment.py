# THIS IS JUST THE MOLECULAR DYNAMICS THAT I USE FOR THE FIRST TRIALS


# ---------------------------------------
import numpy as np
import math
import sys
# ---------------------------------------

# ---------------------------------------
# EQUIVALENT OF GYM FOR MD ENVIRONMENT
# ---------------------------------------

#===============================================================================
# class MDEnv(gym.Env):
# 
#     def __init__(self):
# 
#   def step(self, action):
#     ...
#   def reset(self):
#     ...
#   def render(self, mode='human'):
#     ...
#   def close(self):
#     ...
#===============================================================================

class MD():
    def __init__(self, index=0, N=10, size=10, steps=20, md_type):

        self.dt = 0.2
        self.vel_prey = 0.5
        self.torque_prey = 1.0 / 350.0 * 25.0 # this is Dr * Gamma / kT = 1/350 * 10kT / kT (which is Torque)
        
        self.N = N
        self.size = size
        self.n_MD_steps =steps

        self.Dt = 0.014
        self.Dr = 1.0 / 350.0
        self.Rm = math.sqrt(2*self.Dt/self.dt)
        self.Rr = math.sqrt(2*self.Dr/self.dt)

        self.filexyz='traj'+str(index)+'.xyz'
        self.particles = self.reinitialize_random_for_MD(index)
        self.md_type = md_type
        assert self.md_type in ['group', 'demix'], 'MD type not recognized'
        
# --------------------------
# INITIALIZE RANDOMLY X,Y IN A BOX [-10:10,-10:10] AND THETA [-pi, pi] 
    def reinitialize_random_for_MD(self, index):
        particles = (2*np.random.rand(self.N,3) - [1,1,1]) * [self.size,self.size,np.pi]
        open(self.filexyz, "w") 
        return particles

# ACTUALLY MOVE THE PARTICLES
    def move_all(self, action): # action is vector of ALL actions
        p = self.particles
        g_rand_p = np.random.normal(0,self.Rm,(self.N,3))
        g_rand_p[:,2] =  g_rand_p[:,2] / self.Rm * self.Rr 
        # ACTIVE FORCE OF PREY, IF ACTIVATE
        actvel_p = np.zeros((self.N,3))
        for i in range(self.N):
            move = 0
            #actvel_p[i] = [math.cos(p[i,2])*vel_prey, math.sin(p[i,2])*vel_prey, torque_prey]
            # [0 - go, 1 - stay, 2 - turnclockwise, 3 - turnanticlockwise]
            if ((action[i,2] or action[i,3]) or action[i,0]): move = 1
            actvel_p[i] = [math.cos(p[i,2])*self.vel_prey*move, math.sin(p[i,2])*self.vel_prey*move, (action[i,2]-action[i,3])*self.torque_prey]
        return g_rand_p + actvel_p

        # PRINT TRAJECTORY
    def print_xyz(self):
        p = self.particles
        xyz_file = open(self.filexyz, "a") 
        xyz_file.write(str(self.N)+'\n\n')
        for i in range(self.N):
            xyz_file.write('p '+str(p[i,0])+ ' '+str(p[i,1])+' 0.0 ' + str(p[i,2]) + '\n')
            #xyz_file.write('p '+str(predator[0,0])+ ' '+str(predator[0,1])+' 0.0 ' + str(predator[0,2]) + '\n')

    def get_o_r_group(self):
        p = self.particles
        obs = np.zeros((self.N,5))  # each particle has 5 slices of cone of sight
        rewards = np.zeros((self.N,1))
        value_cone=np.array((1.0, 1.0, 1.0, 1.0, 1.0))
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    n_cone, dist = get_cone_sight(p[i], p[j])
                    if n_cone > -1 and n_cone < 5: 
                        #if dist < 15: 
                        rewards[i]     += 2/(dist/5+10)*value_cone[n_cone]
                        obs[i][n_cone] += 2/(dist/5+10)
            if (obs[i] == 0).all(): rewards[i] -= 2
                    #HERE I SHOULD USE A SATURATING VALUE OF SOMETHING. PERHAPS THE SAME AS IN CLEMEN'S WORK
        return obs, rewards

    def get_o_r_demix(self):
        p = self.particles
        obs = np.zeros((self.N,10))  # each particle has 5 slices of cone of sight
        rewards = np.zeros((self.N,1))
        value_cone=np.array((1.0, 1.0, 1.0, 1.0, 1.0))
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                	  other = (i//(N//2) + j//(N//2))%2
                    n_cone, dist = get_cone_sight(p[i], p[j])
                    if n_cone > -1 and n_cone < 5: 
                        #if dist < 15: 
                        rewards[i]     += 2/(dist/5+10)*value_cone[n_cone] * (-2*other + 1)
                        obs[i][n_cone+5*other] += 2/(dist/5+10)
            if (obs[i,:5] == 0).all(): rewards[i] -= 2
                    #HERE I SHOULD USE A SATURATING VALUE OF SOMETHING. PERHAPS THE SAME AS IN CLEMEN'S WORK
        return obs, rewards

    def get_obs_rewards(self):
        if self.md_type == 'group':
        	   return get_o_r_group()
        elif self.md_type == 'demix':
        	   return get_o_r_demix()
    
    

    def evolve_MD(self, action):
        done = False
        for i in range(self.n_MD_steps):
            dp = self.move_all(action)
            #print(dp)
            self.particles = self.particles + dp * self.dt
            #  predator = predator + dpredator * dt / m / eta
            obs, rewards = self.get_obs_rewards()
            if (obs==0).all(): 
                done = True
        return obs, rewards, done, {}
#
# ------------------------------------
# End of class MD
# ------------------------------------

# CHECK FOR CONE OF SIGHT
def minus_0p5pi_to_1p5pi_mod(theta1):
    return (theta1 + math.pi)%(2*math.pi)-math.pi/2.0

def get_cone_sight(A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    dist = math.sqrt(dx*dx + dy*dy)
    dist_theta = math.atan2(dy, dx)
    rel_theta = ((dist_theta - A[2])/math.pi + 1.0)%(2.0)-0.5
    n_cone = math.floor(rel_theta*5) 
    return int(n_cone), dist
