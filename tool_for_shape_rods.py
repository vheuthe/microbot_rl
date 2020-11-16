#!/usr/bin/env python
import sys
import numpy as np

def spherical_lens(L, dist, Xcenter):
    radius = np.sqrt(Xcenter*Xcenter + L*L/4)
    Psi = np.arcsin(L/2./radius)*2
    Nrod = int(Psi/np.pi*radius)
    rod = np.zeros((2*(Nrod-1),3))
    rod[0,0] = 0
    rod[0,1] = L/2
    rod[-1,0] = 0
    rod[-1,1] = -L/2.
    for i in range(1,Nrod-1):
        Theta = Psi/2. - Psi/(Nrod-1)*i
        rod[i,0] = Xcenter - radius*np.cos(Theta)
        rod[i,1] = radius*np.sin(Theta)
        rod[2*(Nrod-1)-i-1,0] = - rod[i,0]
        rod[2*(Nrod-1)-i-1,1] = rod[i,1]
    rod[:,2] = radius*Psi/(Nrod-1)/2.
    return 2*(Nrod-1), rod

if __name__ == "__main__":
    name_file = sys.argv[1]
    Xcenter = np.float128(sys.argv[2])

    L = np.float128(96)
    dist = 1.3
    Nrod, rod = spherical_lens(L, dist, Xcenter)
    mass = 30 
    inertia = 0
    cm = np.mean(rod, axis=0)
    print(cm.shape)
    for i in range(rod.shape[0]):
        r2 = (rod[i,0]-cm[0])**2+(rod[i,1]-cm[1])**2
        inertia += r2 * mass / Nrod * rod[i,2] / (Nrod * cm[2])
    
    penalty = 0.0

    radius = np.sqrt(Xcenter*Xcenter + L*L/4)
    alpha = np.arcsin(L/2/radius)
    Fnorm = [np.cos(alpha), -L/2/radius]
    Rnorm = [radius * np.cos(alpha) - Xcenter, -L/2]
    penalty = np.abs( mass / inertia * (Fnorm[0]*Rnorm[1] - Fnorm[1]*Rnorm[0]) )

    momo = open(name_file, "w")
    momo.write('{} \n {} {} {} \n'.format(Nrod, mass, penalty, 6.2))
    momo.close()
    momo = open(name_file, "a")
    np.savetxt(momo, rod)
