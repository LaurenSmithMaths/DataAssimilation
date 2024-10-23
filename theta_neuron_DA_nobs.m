%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ensemble Kalman filter for the theta neuron model
% Batched for different number of observed nodes
% Every nth node unobserved
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% Check if on local or server. Local: local == 1, server: local == 0
local = usejava('desktop');

%% Initialise parallel pool

p = gcp('nocreate');

if isempty(p)
    % There is no parallel pool
    %poolsize = 0;
    if (local == 1 )
        parpool(8);
    else
        parpool(16);
    end
    poolsize = p.NumWorkers
else
    % There is a parallel pool of <p.NumWorkers> workers
    poolsize = p.NumWorkers
end

%%

for SWITCH_LOC = [0,1] % standard and localized

dunobs = [2,3,4,5,6,10,12,15,20,30]; 
nr = length(dunobs); 
nr2 = 100; % Number of random realizations
end_rms_theta = zeros(nr,nr2,1);
end_rms_zeta = zeros(nr,nr2,1);

for rr=1:nr
    
     disp(rr);
    disp(dunobs(rr));

tic

parfor rr2 = 1:nr2



%% Set the random seed (so we get reproducible results)
rand_init = 20 + rr2 -1;
rng(rand_init,'twister');

%% Equation parameters
k = 2;
n = 2;

%% theta neuron network connectivity

N = 60;

L = 2*pi;
dx = L/N;

SWITCH_TOP = 2;
% Nearest neighbor ring
if(SWITCH_TOP == 0)
    K = zeros(N,N);
    v = zeros(1,N);
    num_neighbors = 5;
    for i = 1: num_neighbors
        v(i+1) = 1;
        v(N-i+1) = 1;
    end
    K = toeplitz(v);
end

% Coupling from standard model (Laing)
if (SWITCH_TOP == 1)
    K = theta_K(abs([1:N]-[1:N]')*dx);
end

% Some positive coupling, some negative coupling
if (SWITCH_TOP == 2)
    K = zeros(N,N);
    v = zeros(1,N);
    num_neighbors = 3;
    for i = 1: num_neighbors
        v(i+1) = 1;
        v(N-i+1) = 1;
    end
    K = toeplitz(v);
    
    v2 = zeros(1,N);
    num_neighbors_2 = 1;
    for i = 1:num_neighbors_2
        v2(floor(N/2)+i+1) = 1;
        v2(floor(N/2) - i+1) = 1;
        v2(floor(N/2)+1) = 1;
    end
    K2 = toeplitz(v2);
    K = K -0.4* K2;
end

%% Localisation matrix

%SWITCH_LOC = 1;
% No localisation
if (SWITCH_LOC == 0)
    loc_mat = ones(N,N);
end

% Matrix exponential localisation
if (SWITCH_LOC == 1)
    eps = 0.1;
    l0 = 1/(0.55*num_neighbors+0.46); % Approximation based on linear fit of lambda vs r
    l1 = 1.1*l0;
    [expA2, lambda] = ring_loc_param(N,num_neighbors,eps,l0,l1,1e-6);
    loc_mat = mat_exp_loc(abs(K),lambda); % produces the renormalized matexp localization matrix
end

%% intrinsic parameter for each oscillator (eta/zeta)

SWITCH_GAP = 0;
% normal: random
if(SWITCH_GAP == 0)
    mu = -0.4;
    zeta_delta = sqrt(0.1);
    zeta = mu + zeta_delta*randn(N,1);
end

% Lorentzian distribution used by Laing for bump
if(SWITCH_GAP == 1)
mu = -0.4;
zeta_delta = 0.02;
zeta = mu + zeta_delta*tan(pi*(rand(N,1)-1/2));
end

zeta_truth = zeta;

%% Parameters, Matrices etc
% Time step for integration
dt = 0.01;
% Observation time
nt = 1;
dts = nt*dt;
% Number of transient steps
na_trans = 500/dts;
% Number of recorded integration steps
na = 30/dts;
%na = 5/dts;
% Dimension of the system
D = N;

%% Inflation factor
delta = 1 + 0.1*dts;

%% Dimension of observed and unobserved subspaces, and set indices

ind_all   = find(isnan(zeta) == 0);

ind_unobs = [1:dunobs(rr):N];
ind_obs = setdiff(ind_all,ind_unobs);
Ds = length(ind_obs);

Df = D;
Df2 = D-Ds;


% Size of ensemble ms
ms = D+Df+1;
%ms = ms * 10;
% Projections matrix S
w = ones(ms,1)/ms;
S = eye(ms) - w*ones(1,ms);

%% Observation operators
H = zeros(Ds,2*D);
for l=1:Ds
    H(l,ind_obs(l)) = 1;
end

Hs = H(1:Ds,1:D);

%% Variance and mean of observational noise
eta = 0.02;

R = eta^2*eye(Ds);
RI = pinv(R); 

%% Now we build a table of our pseudo-globals, which we pass to many functions
params = struct();
params.D = D;
params.Ds = Ds;
params.Df = Df;
params.ms = ms;
params.k = k;
params.n = n;
params.dt = dt;
params.K = K;
params.zeta = zeta;
params.eta = eta;
params.N = N;

params.locA = [loc_mat loc_mat; loc_mat loc_mat];

%% Create truth and observations   
options = odeset('RelTol',1e-6,'AbsTol',1e-6);

rng(2*rand_init,'twister');

theta0 = 2*pi*rand(N,1);
[T,theta_truth] = ode45(@theta_matrixform,[0:na_trans*dts],theta0,options,K,N,zeta,k,n);

theta0 = mod(theta_truth(end,:),2*pi);
[T,theta_truth] = ode45(@theta_matrixform,[0:dts:na*dts],theta0,options,K,N,zeta,k,n);

frequency = (theta_truth(end,:) - theta_truth(1,:))/(na*dt);

theta_truth = mod(theta_truth,2*pi)';
theta_truth = theta_truth(:,2:end);
time = T(2:end);

%% Observations 
y = mod(Hs*theta_truth + sqrtm(R)*randn(Ds,na),2*pi);

%% Initialisation
Zf = zeros(2*D,ms);
Zf_theta = zeros(ms,D);
Za_theta = zeros(ms,D);
Za_theta_mean = zeros(D,na);
Za_zeta = zeros(ms,D);
Za_zeta_mean = zeros(Df,na);

rms_PertObs_zeta = zeros(1,na);
rms_PertObs_theta = zeros(1,na);

%% EnKF

rng(10*rand_init,'twister');

sigma_theta = 0.2;
sigma_zeta = 0.2;

Zf(1:D,:) = mod(theta_truth(:,1) + sigma_theta*randn(D,ms),2*pi);
Zf(D+1:end,:) = zeta_truth(:) + sigma_zeta*zeta_delta*randn(D,ms);
for m = 1:D
    Zf(m,:) = mod(Zf(m,:) + sigma_theta*randn(),2*pi);
    Zf(D+m,:) = Zf(D+m,:) + sigma_zeta*zeta_delta*randn();
end

ave_Pa = zeros(D+Df,D+Df);
ct=0; % counter

%tic

for j=1:na
    % assimilation
    [Za_bar,Za] = PertObs_loc_circle(Zf,delta,H,R,RI,y(:,j),S,w,params);
    
    if j>na/2
        B_temp = Za - Za_bar*ones(1,ms);
        B_temp(1:D,:) = mod2(B_temp(1:D,:),2*pi,-pi);
        Pa = B_temp*B_temp'/(ms-1); % Covariance matrix of the analysis
        sig = diag(sqrt(Pa)); % SD of each variable
        sigmat = repmat(sig,1,2*D); % SD vector repeated to make a matrix
        sigmat2 = sigmat.*sigmat'; % Pairwise product of SDs, i,j component is sigma_i * sigma_j
        Pa = Pa./sigmat2;
        ave_Pa = ave_Pa + Pa;
        ct = ct + 1;
    end
    
    Za_theta = Za(1:D,:)';
    
    Za_zeta(:,:) = Za(D+1:end,:)';
    
    for m=1:ms
        %[~,Zf_phi] = ode45(@kuramoto_matrixform,[0 dts],Za_phi(m,:),options,A,N,Za_omega(m,:),K);
        %ode45(@theta_matrixform,[0:dt:na*dt],theta0,options,K,N,eta,k,n);
        [~,temp] = ode45(@theta_matrixform,[0 dts],Za_theta(m,:),options,K,N,Za_zeta(m,:)',k,n);
        Zf(1:D,m) = mod(temp(end,:),2*pi)';
        %Zf(1:D,m) = temp(end,:)';
        Zf(D+1:end,m) = Za(D+1:end,m); % dzeta/dt = 0
    end
    
    % storing analysis fields and errors of state and parameters
    Za_theta_mean(:,j) = mod(Za_bar(1:D),2*pi)';
    
    Za_zeta_mean(:,j) = Za_bar(D+1:end)';
    
    rms_PertObs_zeta(j) = norm(Za_zeta_mean(:,j) - zeta_truth(:), 'fro') ...
        / sqrt(N);
    
    
    % RMS error
    rms_PertObs_theta(j)   = norm(mod2(Za_theta_mean(:,j)-theta_truth(:,j),2*pi,-pi), 'fro') / sqrt(N);
    
    
end

end_rms_theta(rr,rr2) = rms_PertObs_theta(end);
end_rms_zeta(rr,rr2) = rms_PertObs_zeta(end);

end

toc

end

%%
if (SWITCH_LOC == 1)
    save('DA_theta_ring_nobs_loc_1.mat')
else
    save('DA_theta_ring_nobs_no_loc_1.mat')
end

end

%% Figures (after running both nobs programs)
a1 = load('DA_theta_ring_nobs_no_loc_1.mat');
a2 = load('DA_theta_ring_nobs_no_loc_2.mat');
b1 = load('DA_theta_ring_nobs_loc_1.mat');
b2 = load('DA_theta_ring_nobs_loc_2.mat');

%%
N = 60;
dunobs = [2,3,4,5,6,10,12,15,20,30];
nunobs1 = N./dunobs;
nobs1 = N - nunobs1;

dobs = [3,4,5,6,10,12,15,20,30]; 
nobs2 = flip(N./dobs);
nunobs2 = N - nobs2;

nobs = [nobs2, nobs1];

%%
h = figure(2);
plot(nobs/N,[flip(median(a2.end_rms_theta,2)); median(a1.end_rms_theta,2)],'--bx','LineWidth',2,'MarkerSize',10)
hold on
plot(nobs/N,[flip(median(a2.end_rms_zeta,2)); median(a1.end_rms_zeta,2)],'-bo','LineWidth',2,'MarkerSize',10)
hold on
plot(nobs/N,[flip(median(b2.end_rms_theta,2)); median(b1.end_rms_theta,2)],'--rx','LineWidth',2,'MarkerSize',10)
hold on
plot(nobs/N,[flip(median(b2.end_rms_zeta,2)); median(b1.end_rms_zeta,2)],'-ro','LineWidth',2,'MarkerSize',10)
hold off
set(gca,'YScale','log')
set(gca,'FontSize',18)
xlabel('Fraction of observed nodes','Interpreter','latex','FontSize',20)
ylabel('RMS error','Interpreter','latex','FontSize',20)
legend('$E_\phi$ standard','$E_\zeta$ standard','$E_\phi$ localized','$E_\zeta$ localized','Interpreter','latex','FontSize',18,'Location','southwest')



