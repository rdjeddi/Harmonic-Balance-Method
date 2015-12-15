function utsav()
% Utsav Parmar 
% U00768046
clear all
clc
close all

%ddx + dx + x = cos(w*t)

N=99;
t=linspace(0,2*1*pi, N+1);
t = t(1:end-1)';
T=2;
w=2*pi/T;
omega =[0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]'+0.0000001;
Omega=w*omega;
f=cos(2.*t);
F=fft(f);
t_eval = linspace(0,2*pi,100);
X_analytical = (-w*F-sqrt(1-F.^2).*(1-w^2))/(w^2-w^4-1);
x0 = 0*ones(N,1);
X1 = fminsearch(uvfun(),F);
function RES = uvfun(x0)
 dX=Omega.*F;
 dx=ifft(dX);
 dotx=ifft(Omega.*F);
 ddx = ifft(-Omega.^2.*F);
 x0=ddx./Omega.^2;
 error = (ddx + dotx + x0 - f);
 RES = @(F) sum((error).^2);
end
 plot(t,-w*f-sqrt(1-f.^2).*(1-w^2)/(w^2-w^4-1),'r*',t,-w*ifft(F)-sqrt(1-ifft(F).^2).*(1-w^2)/(w^2-w^4-1), 'b-');
 xlabel('Time (seconds)');
 ylabel('x(t)');
 title('Linear damped case')
 legend('Analytical Solution','Harmonic Balance Solution')
end
