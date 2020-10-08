import numpy as np

def spherical_lens(L, Nrod, radius):
    rod = np.zeros((2*(Nrod-1),3))
    Xcenter = np.sqrt(radius*radius-L*L/4)
    Psi = np.arcsin(L/2./radius)
    rod[0,:] = [0,L/2,3.6]
    rod[-1,:] = [0,-L/2,3.6]
    for i in range(1,Nrod-1):
        Theta = Psi - 2*Psi/(Nrod-1)*i
        rod[i,0] = Xcenter - radius*np.cos(Theta)
        rod[i,1] = radius*np.sin(Theta)
        rod[2*(Nrod-1)-1-i,0] = - rod[i,0]
        rod[2*(Nrod-1)-1-i,1] = rod[i,1]
    for i in range(2*(Nrod-1)):
        rod[i,2] = (L/2.-abs(rod[i,1]))*2/L*0 + 6.2
    return rod