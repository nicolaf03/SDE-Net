"""
Created on Fri Aug 14 00:06:57 2020
@author: Justin Yu
Implementation of Fractional Brownian Motion, Hosking's method.
"""
import numpy as np
import matplotlib.pyplot as plt

def hosking(T, N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k, H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))
    
    X = [np.random.standard_normal()]
    mu = [gamma(1,H)*X[0]]
    sigsq = [1 - (gamma(1,H)**2)]
    tau = [gamma(1,H)**2]
    
    d = np.array([gamma(1,H)])
    
    for n in range(1, N):
        
        F = np.rot90(np.identity(n+1))
        c = np.array([gamma(k+1,H) for k in range(0,n+1)])
                
        # sigma(n+1)**2
        s = sigsq[n-1] - ((gamma(n+1,H) - tau[n-1])**2)/sigsq[n-1]
        
        # d(n+1)
        phi = (gamma(n+1,H) - tau[n-1])/sigsq[n-1]
        d = d - phi*d[::-1]
        d = np.append(d, phi)        
        
        # mu(n+1) and tau(n+1)
        Xn1 = mu[n-1] + sigsq[n-1]*np.random.standard_normal()
        
        X.append(Xn1)
        sigsq.append(s)
        mu.append(d @ X[::-1])
        tau.append(c @ F @ d)
    
    fBm = np.cumsum(X)*(N**(-H))    
    return (T**H)*fBm

"""
H = 1 / 2
a1 = [0.0]
for i in range(0, 200):
  a1.append(hosking(1, 1, H))

a = hosking(1, 200, H);#plt.plot(np.cumsum((a))*np.sqrt(1/200),lw=2)
a = np.insert(a, 0, 0.0)

plt.plot(a, 'k', lw=2)
plt.plot(a1, 'r', lw=2)
plt.show()
"""