%%%%%%%%
%Equation file for ODE 45 describing the Kuramoto system, 
%    dy(i)/dt = w(i) + K/N* \sum A_{ij} sin(y_j-y_i) 
%%%%%%%

function dy = kuramoto_matrixform(t,y,A,N,w,K)

%dy=zeros(N,1);
%%% Calculate each of the system's N equations
PHI = repmat(y',N,1);

S = PHI - PHI';

dy = w' + (K/N)*sum(A.*sin(S),2);
