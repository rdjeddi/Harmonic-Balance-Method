__author__ = 'DanielClarkJR'
import pylab as pl
import numpy as np
import scipy.optimize as sci
import scipy.integrate as sp

# this code is the nonlinear case of \dotdot{x} + \dot{x} + x - x**3 = cos(2*t)

# Plotting some solutions
def solve_hw2(max_time=5,x0 = 1, v0 = 1):
    def hw2_deriv(x1_x2,t):
        x1, x2 = x1_x2
        return [x2, -x2-x1+0*x1**3+np.cos(2*t)]
    t = np.linspace(0, max_time, int(2000*max_time))
    x_t = sp.odeint(hw2_deriv, [x0,v0], t)
    return t, x_t

t, x_t = solve_hw2(max_time=10*np.pi, x0 = 0, v0 = 0)

pl.plot(t,x_t[:,0])
pl.xlabel('$t$', fontsize=20)
pl.ylabel('$x(t)$', fontsize=20)
pl.grid()

# this is method 1
N = 99
t = np.linspace(0, 10*np.pi, N+1)    # time samples of forcing function
t = t[0:-1]                         # Removing the extra sample
f = np.cos(2*t)                     # My forcing function
T = t[-1]

# this is the Dr. Slater method, this will work with nonlinear functions
xbar = 10*f

def FUNCTION(xbar):
    N = len(xbar)
    Xbar = np.fft.fft(xbar)
    omega = np.fft.fftfreq(N, T/(2*np.pi*N) )# + 0.0000001 # list of frequencies
    dotxbar = np.fft.ifft(np.multiply((1j*omega),Xbar))
    dotdotxbar = np.fft.ifft(np.multiply((1j*omega)**2,Xbar))
    R = dotdotxbar + dotxbar + xbar - 0*xbar**3 - f
    R = R**2
    R = np.sum(R)
    return R

optimizedResults = sci.minimize(FUNCTION, xbar, method='SLSQP')
xbar = optimizedResults.x

print(optimizedResults)
print(xbar)

pl.scatter(t,xbar)
pl.show()
