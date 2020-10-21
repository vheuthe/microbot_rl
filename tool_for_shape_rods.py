#!/usr/bin/env python
import sys
import numpy as np

def spherical_lens(L, dist, Xcenter):
    radius = np.sqrt(Xcenter*Xcenter + L*L/4)
    Psi = np.arcsin(L/2./radius)*2
    Nrod = int(Psi/np.pi*radius)
    rod = np.zeros((2*Nrod,3))
    for i in range(0,Nrod):
        Theta = Psi/2. - Psi/(Nrod-1)*i
        rod[i,0] = Xcenter - radius*np.cos(Theta) + 0.1
        rod[i,1] = radius*np.sin(Theta)
        rod[2*Nrod-i-1,0] = - rod[i,0]
        rod[2*Nrod-i-1,1] = rod[i,1]
    for i in range(2*Nrod):
        scale = 1./6.2
        for j in range(2*Nrod):
            if (i != j):
                d = np.sqrt((rod[i,0]-rod[j,0)**2 + (rod[i,1]-rod[j,1)**2 ) 
            scale += 1/(6.2 + d)
        rod[i,2] = (6.2 + d) / (2*6.2 + d)
    return 2*Nrod, rod

if __name__ == "__main__":
    name_file = sys.argv[1]
    Xcenter = np.float128(sys.argv[2])

    L = np.float128(96)
    dist = 1.3
    Nrod, rod = spherical_lens(L, dist, Xcenter)
    mass = 30 
  
    cm = np-mean(rod, axis=1)
    for i in rod.shape[0]:
        r2 = (rod[i,0]-cm[0])**2+(rod[i,1]-cm[1])**2
        inertia += r2 * mass / Nrod * rod[2]/cm[2]/Nrod
    
    penalty = L/2 / inertia
    
    momo = open(name_file, "w")
    momo.write('{} \n {} {} {} {} \n'.format(Nrod, mass, penalty, 6.2))
    momo.close()
    momo = open(name_file, "a")
    np.savetxt(momo, rod)