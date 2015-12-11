__author__ = 'DanielClarkJR'
import numpy as np
import pylab as pl
import scipy.optimize as sci
import scipy.integrate as sp

#  ddx - mu*(1 - x**2)*dx + x = A*cos(w*t)


def solve_sdof(timeStep=10.0,mu = 9,A = 1 ,w = 3, x0 = 0, v0 = 0):
    def sdof_deriv(x1_x2, t):
        """Compute the time-derivative of a SDOF system."""
        x1, x2 = x1_x2

        return [x2, A*np.cos(w*t)+mu*(1-x1**2)*x2-x1]

    x0i=((x0, v0))
    t = np.linspace(0, 100*timeStep, int(500*timeStep))
    x_t = sp.odeint(sdof_deriv, x0i, t)
    x, v = x_t.T

    return t, x, v

def HarmonicBalanceMethod(N=41,A = 9, mu = 1, w = 3):

    time = np.linspace(0, 2*np.pi, N)       # time samples of forcing function
    time = time[0:-1]                       # Removing the extra sample
    omega = np.fft.fftfreq(N-1, 1/(N-1) )   # FFT frequencies
    f = A*np.cos(w*time)                      # My forcing function
    xbar = A*np.cos(w*time)                   # Guess

    def FUNCTION(xbar):
        Xbar = np.fft.fft(xbar)
        dotxbar = np.fft.ifft(np.multiply((1j*omega),Xbar))
        dotdotxbar = np.fft.ifft(np.multiply(-omega**2,Xbar))
        R = dotdotxbar - mu*(1-xbar**2)*dotxbar + xbar - f
        return np.sum(np.abs(R**2))

    optimizedResults = sci.minimize(FUNCTION, xbar, method = 'BFGS', options={'maxiter':50000, 'disp':True})
    xbar = optimizedResults.x
    print(FUNCTION(xbar))
    harmonicTime = time    # shift to steady state time for plotting

    return harmonicTime, xbar

A = 1.2
w = 3*np.pi
mu = 1.2
Period = 2*np.pi/w

num = 91

t,x,v = solve_sdof(timeStep=Period, x0 = 0, v0 = 0,A = A,mu=mu, w=w)
time,xbar = HarmonicBalanceMethod(N = num,A=A,mu=mu,w=w,)


pl.figure(figsize=(15,10))
pl.plot(t,x)
pl.plot(time+11.2*np.pi,xbar,lw=3)
pl.legend(['Integrated Solution','Harmonic Balance Method Response'])
pl.xlabel('Time(s)')
pl.title('Comparison of Results for the Van der Pol Oscillator')
pl.show()
