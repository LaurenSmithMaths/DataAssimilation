function [B, lambda] = ring_loc_param(N,r,eps,lambda0,lambda1,accuracy)
% determine the localisation parameter for expm(lambda*A)
%   N = number of nodes
%   r = radius of connectivity for ring, degree = 2*r
%   eps = cutoff value

% Define ring adjacency matrix
A = zeros(N,N);
v = zeros(1,N);
num_neighbors = r;
for i = 1: num_neighbors
    v(i+1) = 1;
    v(N-i+1) = 1;
end
A = toeplitz(v);

l0 = lambda0; % starting guess 1
B0 = mat_exp_loc(A,l0);
diff0 = B0(1,2*r+2) - eps;
l1 = lambda1; % starting guess 2
B1 = mat_exp_loc(A,l1);
diff1 = B1(1,2*r+2) - eps;
ct = 0;
while( abs(diff1) > accuracy && ct < 20 )
    l2 = (l0*diff1 - l1*diff0)/(diff1 - diff0); % secant method
    B2 = mat_exp_loc(A,l2);
    diff2 = B2(1,2*r+2) - eps;
    
    % Update
    l0 = l1;
    B0 = B1;
    diff0 = diff1;
    
    l1 = l2;
    B1 = B2;
    diff1 = diff2;
    
    ct = ct + 1;
end

B = B1;
lambda = l1;
    



end

