function utsav()
% Utsav Parmar 
% U00768046
clear all
clc
close all

%ddx + x = cos(2*t)

N=100;
t=linspace(0,2*1*pi, N+1);
t = t(1:end-1)';
T=2;
w=2*pi/T;
omega =[0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]'+0.0000001;
Omega=w*omega;
f=cos(2.*t);
F=fft(f);
t_eval = linspace(0,2*pi,100);
X_analytical = (cos(2*t_eval)/(1-w.^2));
%plot(t_eval,X_analytical, '-r');
x0 = 0*ones(N,1);

X1 = fminsearch(uvfun(),F);

function RES = uvfun(x0)
 dX=Omega.*F;
 dx=ifft(dX);
 ddx = ifft(-Omega.^2.*F);
 x0=ddx./Omega.^2;
 error = (ddx + x0 - f);
 RES = @(F) sum((error).^2);
end
 plot(t_eval,X_analytical,'r*',t,ifft(X1)/(1-w.^2), 'r-');
 xlabel('Time (seconds)');
 ylabel('x(t)');
 title('Linear undamped case')
 legend('Analytical Solution','Harmonic Balance Solution')
end
