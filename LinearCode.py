__author__ = 'DanielClarkJR'
import pylab as pl
import numpy as np
import scipy.optimize as sci

# I am solving the equation dotdotX + X = cos(2*t)


# this is method 1
N = 17
t = np.linspace(0, 2*np.pi, N+1)    # time samples of forcing function
t = t[0:-1]                         # Removing the extra sample
f = np.cos(2*t)                     # My forcing function
F = np.fft.fft(f)
omega = np.fft.fftfreq(N, 1/N) + 0.0000001 # list of frequencies
X = np.divide(F, 1 - omega**2)
x = np.fft.ifft(X)

t_eval = np.linspace(0,2*np.pi,100)
X_analytical = -(np.cos(2*t_eval)/3)
pl.plot(t_eval, X_analytical)
#pl.scatter(t,x)
#pl.show()

# this is the Dr. Slater method, this will work with nonlinear functions
xbar = f*0 + np.cos(2*t)

def FUNCTION(xbar):
    N = len(xbar)
    Xbar = np.fft.fft(xbar)
    omega = np.fft.fftfreq(N, 1/N) + 0.0000001 # list of frequencies
    dotdotxbar = np.fft.ifft(np.multiply((1j*omega)**2,Xbar))
    R = np.sum(np.real(np.abs(dotdotxbar + xbar - f)))
    return R

optimizedResults = sci.minimize(FUNCTION, xbar, method='SLSQP')
xbar = optimizedResults.x

#optimizedResults = sci.fmin(FUNCTION, xbar, args=(), xtol=0.0000001, ftol=0.0000001,maxiter=10000, maxfun= 100000)
#print(optimizedResults)
#xbar = optimizedResults

# func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None


print(optimizedResults)
print(xbar)

# pl.plot(t_eval, X_analytical)
pl.scatter(t,xbar)
pl.show()
