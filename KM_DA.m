%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ensemble Kalman filter for the Kuramoto model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set the random seed (so we get reproducible results)
rand_init = 20;
rng(rand_init);

%% Equation parameters
num_neigh = 3; % number of neighbours in the ring topology

K = 80/num_neigh; % coupling strength 

N = 50; % number of nodes in the network

obs_frac = 0.7; % Fraction of observed phases

%% Switches for selecting options
SWITCH_TOP = 3; % network topology, 3 = ring
SWITCH_GAP = 3; % native frequencies, 3 = normally distributed
SWITCH_LOC = true; % select localized or standard EnKF, true = localized
SWITCH_TRANS = false; % whether to discard an initial transient in the integration. false = keep transient

%% Network topology

num_con_components = 0; % number of connected components

while num_con_components ~= 1

% All-to-All coupling
if(SWITCH_TOP == 0)
    A = ones(N)-eye(N);
end
% Erdos-Renyi Graph
if(SWITCH_TOP == 1)
    p = 0.05; %Probability 2 nodes are connected
    p = 0.1;

    A = binornd(1,p,N,N);
    A(1:N+1:N*N) = 0;
    A = A - tril(A);
    A = A + triu(A,1).';
end
%Nearest neighbor line (not ring)
if(SWITCH_TOP == 2)
    A = zeros(N,N);
    for j=1:N-1
        A(j,j+1) = 1;
        A(j+1,j) = 1;
    end
end
%Nearest neighbor ring
if(SWITCH_TOP == 3)
    A = zeros(N,N);
    v = zeros(1,N);
    num_neighbors = num_neigh;
    for i = 1: num_neighbors
        v(i+1) = 1;
        v(N-i+1) = 1;
    end
    A = toeplitz(v);
end
% Barabasi-Albert scale-free network
if(SWITCH_TOP == 4)
    m1 = 1;
    k = 6; % target mean degree
    m2 = k-m1;
    seed_net =[0 1 0 0 1;1 0 0 1 0;0 0 0 1 0;0 1 1 0 0;1 0 0 0 0];
    rng_seed = 3*rand_init;
    A = SFNG_modified(N,m1,m2,seed_net,rng_seed);
    degrees = sum(A);
end

degrees = sum(A);
L = diag(degrees) - A; % Graph Laplacian
num_con_components = length(find(abs(eig(L)) < 10^-10));
end

mean_degree = mean(degrees);
num_neighbors = mean_degree/2; % Equivalent number of neighbours for a ring topology

%% Matrix exponential localization
eps = 0.1; % epsilon used for localization
if (abs(round(num_neighbors) - num_neighbors)<1E-12)
    % num_neighbors is an integer, no need for interpolation
    l0 = 1/(0.55*num_neighbors+0.46); % Approximation based on linear fit of lambda vs r
    l1 = 1.1*l0;
    [temp, lambda] = ring_loc_param(N,round(num_neighbors),eps,l0,l1,1e-6);
    % the output lambda is the lambda used for the matrix exp localization
else
    % num_neighbors is not an integer, we need to interpolate to find
    % lambda
    r_l = floor(num_neighbors);
    r_u = ceil(num_neighbors);
    
    % Work with r_l first
    l0 = 1/(0.55*r_l+0.46); % Approximation based on linear fit of lambda vs r
    l1 = 1.1*l0;
    [temp, lambda_l] = ring_loc_param(N,r_l,eps,l0,l1,1e-6);
    
    % Repeat for r_u
    l0 = 1/(0.55*r_u+0.46); % Approximation based on linear fit of lambda vs r
    l1 = 1.1*l0;
    [temp, lambda_u] = ring_loc_param(N,r_u,eps,l0,l1,1e-6);
    
    % Now interpolate between them, based on 1/lambda = m*r + c
    m = (1/lambda_u - 1/lambda_l)/(r_u - r_l);
    c = 1/lambda_l - m*r_l;
    lambda = 1/(m*num_neighbors + c);
    % the output lambda is the lambda used for the matrix exp localization
end

loc_mat = mat_exp_loc(A,lambda);

%% Native frequencies 
rng(3*rand_init + 1);
% uniform: random
if(SWITCH_GAP == 1)
    omega = -1 + 2*rand(1,N);
    omega = sort(omega);
end
% uniform: equiprobable
if(SWITCH_GAP == 2)
    a = 1;
    i_vec = cumsum(ones(1,N));
    argument = 2*(i_vec-(N+1)/2)/(N-1);
    omega = argument.^a;
end
% normal: random
if(SWITCH_GAP == 3)
    sigma_omega = sqrt(0.1);
    omega = sigma_omega*randn(1,N);
    %omega = sort(omega);
    omega = omega - mean(omega);
end
% normal: equiprobable
if(SWITCH_GAP == 4)
    dN = 1/(N+1);
    i_vec = dN:dN:1-dN;
    sigma_omega = sqrt(0.02);
    omega = icdf('Normal',i_vec,0,sigma_omega);
end
% bimodal normal: random
if(SWITCH_GAP == 5)
    sigma_omega = sqrt(0.1);
    omega = sigma_omega*randn(1,N);
    domega = 10 * sigma_omega;
    N1 = randi([floor(N/3),ceil(2*N/3)],1,1);
    N2 = N-N1;
    ind1 = sort(randperm(N,N1));
    ind2 = setdiff([1:N],ind1);
    omega(ind1) = omega(ind1) + domega/2;
    omega(ind2) = omega(ind2) - domega/2;
end

%% Dimension of the observed and unobserved subspaces, and set indices

% Dimension of the system
D = N;

ind_all   = find(isnan(omega) == 0);

% Dimension of observed subspace
Ds = floor(obs_frac*D);
ind_obs = sort(randperm(D,Ds)); % Random selection of observed phases

% Dimension of unobserved x subspace
Df = D;
Df2 = D-Ds;
ind_unobs = setdiff(ind_all,ind_obs);

% Size of ensemble ms
ms = D+Df+1;

% Projections matrix S
w = ones(ms,1)/ms;
S = eye(ms) - w*ones(1,ms);

%% Observation operators
% Observation operator H
H = zeros(Ds,2*D);
for l=1:Ds
    %H(l,(l-1)*dobs+1) = 1;
    H(l,ind_obs(l)) = 1;
end

Hs = H(1:Ds,1:D);

%% Variance and mean of observational noise
eta = 0.02;

R = eta^2*eye(Ds);
RI = pinv(R); 

%% Inflation factor
delta = 1.001;

%% Integration parameters
% Time step for integration
dt = 0.01;
% Observation time
nt = 1;
dts = nt*dt;
% Number of recorded integration steps
na = 10/dts;

%% Structure with pseudo-globals, which we pass to many functions
params.D = D;
params.Ds = Ds;
params.Df = Df;
params.ms = ms;
params.K = K;
params.dt = dt;
params.A = A;
params.omega = omega;
params.sigma_omega = sigma_omega;
params.N = N;

if (SWITCH_LOC == false)
    % No localization, standard EnKF
    params.locA = ones(2*D,2*D);
else
    % EnKF with network specific localization
    params.locA = [loc_mat loc_mat; loc_mat loc_mat];
end

%% Create truth and observations   
options = odeset('RelTol',1e-6,'AbsTol',1e-6);

omega_truth = omega;

rng(2*rand_init);

phi0 = 2*pi*rand(1,N); % Uniformly random on [0,2pi]


if (SWITCH_TRANS)
    [T,temp] = ode45(@kuramoto_matrixform,[0:dts:20*na*dts],phi0,options,A,N,omega_truth,K);
    phi0 = mod(temp(end,:),2*pi);
end

[T,phi_truth] = ode45(@kuramoto_matrixform,[0:dts:na*dts],phi0,options,A,N,omega_truth,K);
phi_truth = mod(phi_truth,2*pi)';
phi_truth = phi_truth(:,2:end);
time = T(2:end);

%% Observations 
y = mod(Hs*phi_truth + sqrtm(R)*randn(Ds,na),2*pi);

%% Initialisation of DA matrices
Zf = zeros(2*D,ms);
Zf_phi = zeros(ms,D);
Za_phi = zeros(ms,D);
Za_phi_mean = zeros(D,na);
Za_omega = zeros(ms,D);
Za_omega_mean = zeros(Df,na);

rms_omega = zeros(1,na);
rms_phi = zeros(1,na);

%% EnKF

% Initial ensemble
rng(4*rand_init+1);

sigma_phi = 0.1;

sigma_omega_2 = 0.1;

Zf(1:D,:) = mod(phi_truth(:,1) + sigma_phi*randn(D,ms),2*pi);
Zf(D+1:end,:) = omega_truth(:) + sigma_omega_2*sigma_omega*randn(D,ms);
for m = 1:D
    Zf(m,:) = mod(Zf(m,:) + sigma_phi*randn(),2*pi);
    Zf(D+m,:) = Zf(D+m,:) + sigma_omega_2*sigma_omega*randn();
end

init_omega_mean = mean(Zf(D+1:end,:),2);
init_err_omega = norm(init_omega_mean - omega_truth', 'fro') ...
    / sqrt(Df);

init_phi_mean = angle(mean(exp(1i*Zf(1:D,:)),2));
init_err_phi = norm(mod2(init_phi_mean-phi0',2*pi,-pi), 'fro') / sqrt(D);

ave_Pf = zeros(D+Df,D+Df);
ct=0; % counter
for j=1:na
    % assimilation
    [Za_bar,Za,Pf] = PertObs_loc_circle_out_Pf(Zf,delta,H,R,RI,y(:,j),S,w,params);
    
    % Averaging the Pf across the DA process
    if j>na/10
        sigf = diag(sqrt(Pf)); % SD of each variable
        sigmatf = repmat(sigf,1,2*D); % SD vector repeated to make a matrix
        sigmatf2 = sigmatf.*sigmatf'; % Pairwise product of SDs, i,j component is sigma_i * sigma_j
        Pf = Pf./sigmatf2;
        ave_Pf = ave_Pf + Pf;
        ct = ct + 1;
    end
    
    % forecasting
    Za_phi = Za(1:D,:)';
    
    Za_omega(:,:) = Za(D+1:end,:)';
    
    for m=1:ms
        [~,temp] = ode45(@kuramoto_matrixform,[0 dts],Za_phi(m,:),options,A,N,Za_omega(m,:),K);
        Zf(1:D,m) = mod(temp(end,:),2*pi)'; % phases
        Zf(D+1:end,m) = Za(D+1:end,m); % domega/dt = 0
    end
    
    % storing analysis fields and errors of state and parameters
    Za_phi_mean(:,j) = mod(Za_bar(1:D),2*pi)';
    
    Za_omega_mean(:,j) = Za_bar(D+1:end)';
    
    % RMS errors in omega and phi
    rms_omega(j) = norm(Za_omega_mean(:,j) - omega_truth(:), 'fro') / sqrt(Df);
    rms_phi(j)   = norm(mod2(Za_phi_mean(:,j)-phi_truth(:,j),2*pi,-pi), 'fro') / sqrt(D);
    
end

ave_Pf = ave_Pf/ct;
