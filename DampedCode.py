__author__ = 'DanielClarkJR'

import pylab as pl
import numpy as np
import scipy.optimize as sci

# This equation has a damped term
# I am solving the equation dotdotX + dotX + X = cos(2*t)

# this is method 1
N = 40
t = np.linspace(0, 2*np.pi, N+1)    # time samples of forcing function
t = t[0:-1]                         # Removing the extra sample
f = np.cos(2*t)                     # My forcing function

t_eval = np.linspace(0,2*np.pi,100)
X_analytical = (2/13)*np.sin(2*t_eval) - (3/13)*np.cos(2*t_eval)

# this is the Dr. Slater method, this will work with nonlinear functions
xbar = f

def FUNCTION(xbar):
    N = len(xbar)
    Xbar = np.fft.fft(xbar)
    omega = np.fft.fftfreq(N, 1/N) + 0.0000001 # list of frequencies
    dotxbar = np.fft.ifft(np.multiply((1j*omega),Xbar))
    dotdotxbar = np.fft.ifft(np.multiply((1j*omega)**2,Xbar))
    R = dotdotxbar + dotxbar + xbar - f
    R = R**2
    R = np.sum(R)
    return R

optimizedResults = sci.minimize(FUNCTION, xbar, method='SLSQP')
xbar = optimizedResults.x

print(optimizedResults)
print(xbar)

pl.plot(t_eval, X_analytical)
pl.scatter(t,xbar)
pl.show()
