%%%%%%%%
%Equation file for ODE 45 describing the theta neuron system
%%%%%%%

function dy = theta_matrixform(t,y,K,N,eta,k,n)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%size(y)

L = 2*pi;

dx = L/N;

P_vec = theta_P(y,n);

%size(P_vec)

I_vec = dx*sum(K.*P_vec)';

%size(I_vec)

dy = 1 - cos(y) + (1+cos(y)).*(eta + k*I_vec);

%size(dy)


end

