function P = theta_P(theta,n)
% P function for theta neurons
%   Detailed explanation goes here

P = (2^n * factorial(n)^2)/(factorial(2*n))*(1-cos(theta)).^n;

end

