%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ensemble Kalman filter for the theta neuron model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set the random seed (so we get reproducible results)
rand_init = 20;
rng(rand_init);

%% Equation parameters
k = 2;
n = 2;

%% Switches for selecting options
SWITCH_TOP = 2; % network topology, 2 = some positive, some inhibitive, ringlike
%SWITCH_LOC = 1; % localization, 1 = localized EnKF
SWITCH_GAP = 0; % Intrinsic firing parameters, 0 = normally distributed

for SWITCH_LOC = [0,1]

%% theta neuron network connectivity

N = 50;

L = 2*pi;
dx = L/N;

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

%%
figure(99)
imagesc(K)
pbaspect([1,1,1])
colorbar
set(gca,'FontSize',14)
title('coupling matrix')

%% Localisation matrix

% No localisation
if (SWITCH_LOC == 0)
    loc_mat = ones(N,N);
end

% Matrix exponential localisation
if (SWITCH_LOC == 1)
    eps = 0.1; % epsilon used for localization
    l0 = 1/(0.55*num_neighbors+0.46); % Approximation based on linear fit of lambda vs r
    l1 = 1.1*l0;
    [expA2, lambda] = ring_loc_param(N,num_neighbors,eps,l0,l1,1e-6); % determines lambda
    loc_mat = mat_exp_loc(abs(K),lambda); % produces the renormalized matexp localization matrix
    
    figure(98)
    imagesc(loc_mat)
    pbaspect([1,1,1])
    colorbar
    set(gca,'FontSize',14)
    title('log10 of matrix exp')
end




%% intrinsic parameter for each oscillator (eta/zeta)

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
% Dimension of the system
D = N;

%% Inflation factor
delta = 1 + 0.1*dts;

%% Dimension of observed and unobserved subspaces, and set indices

ind_all   = find(isnan(zeta) == 0);

% Dimension of observed subspace
Ds = floor(0.7*D);
ind_obs = sort(randperm(N,Ds));
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

rng(2*rand_init);

theta0 = 2*pi*rand(N,1);
% transient that is discarded
[T,theta_truth] = ode45(@theta_matrixform,[0:na_trans*dts],theta0,options,K,N,zeta,k,n);

% integration after the transient, on the attractor
theta0 = mod(theta_truth(end,:),2*pi);
[T,theta_truth] = ode45(@theta_matrixform,[0:dts:na*dts],theta0,options,K,N,zeta,k,n);

% effective frequencies of each oscillator
frequency = (theta_truth(end,:) - theta_truth(1,:))/(na*dt);

theta_truth = mod(theta_truth,2*pi)';
theta_truth = theta_truth(:,2:end);
time = T(2:end);


%%
figure(1)
plot([1:N]*dx,theta_truth(:,end),'.')
set(gca,'FontSize',14)
xlabel('$x$','Interpreter','latex','FontSize',18)
ylabel('$\theta$','Interpreter','latex','FontSize',18)

%%
figure(2)
plot([1:N]*dx,frequency,'.')
set(gca,'FontSize',14)
xlabel('$x$','Interpreter','latex','FontSize',18)
ylabel('frequency','Interpreter','latex','FontSize',18)

%%
figure(3)
imagesc(theta_truth)
colorbar
colormap('hsv')

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

tic

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
        [~,temp] = ode45(@theta_matrixform,[0 dts],Za_theta(m,:),options,K,N,Za_zeta(m,:)',k,n);
        Zf(1:D,m) = mod(temp(end,:),2*pi)';
        Zf(D+1:end,m) = Za(D+1:end,m); % dzeta/dt = 0
    end
    
    % storing analysis fields and errors of state and parameters
    Za_theta_mean(:,j) = mod(Za_bar(1:D),2*pi)';
    
    Za_zeta_mean(:,j) = Za_bar(D+1:end)';
    
    % RMS errors in zeta and theta
    rms_PertObs_zeta(j) = norm(Za_zeta_mean(:,j) - zeta_truth(:), 'fro') ...
        / sqrt(N);
    rms_PertObs_theta(j)   = norm(mod2(Za_theta_mean(:,j)-theta_truth(:,j),2*pi,-pi), 'fro') / sqrt(N);
    
    
end

toc

ave_Pa = ave_Pa/ct;

if (SWITCH_LOC == 1)
    save("theta_DA_ring_0.7_observed_loc.mat");
else
    save("theta_DA_ring_0.7_observed_no_loc.mat");
end

end

%% Figures

a = load("theta_DA_ring_0.7_observed_loc.mat");
b = load("theta_DA_ring_0.7_observed_no_loc.mat");

%% theta_i vs t
figure(3)
clims = [0 2*pi];
imagesc([a.dts,a.na*a.dts],[1,a.N],a.theta_truth,clims)
pbaspect([1,1,1])
c = colorbar;
colormap('hsv')
xlim([0 a.na*a.dts])
set(gca,'FontSize',18)
yticks([1,10,20,30,40,50])
set(gca,'TickLength',[0 0])
xlabel('$t$','Interpreter','latex','FontSize',20)
ylabel('$i$','Interpreter','latex','FontSize',20)
c.Ticks = [0:pi/2:2*pi];
c.TickLabels = {'$0$','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'};
c.TickLabelInterpreter = 'latex';
c.FontSize = 18;

%% RMS errors
h = figure(12);
plot(b.time,(b.rms_PertObs_theta),'--b','LineWidth',2)
hold on
plot(b.time,(b.rms_PertObs_zeta),'-b','LineWidth',2)
hold on
plot(a.time,(a.rms_PertObs_theta),'--r','LineWidth',2)
hold on
plot(a.time,(a.rms_PertObs_zeta),'-r','LineWidth',2)
hold off
set(gca,'YScale','log')
set(gca,'FontSize',18)
ylim([1E-3 1E0])
xlabel('$t$','Interpreter','latex','FontSize',20)
ylabel('RMS error','Interpreter','latex','FontSize',20)
legend('$E_\phi$ standard','$E_\zeta$ standard','$E_\phi$ localized','$E_\zeta$ localized','Interpreter','latex','FontSize',18,'Location','southwest')

%% Residuals in theta
h = figure(13);

%scatter(1:a.N,mod2(a.phi_truth(:,end)-a.Za_phi_mean(:,end),2*pi,-pi),[],'r')
b_obs_mean = median(abs(mod2(b.theta_truth(b.ind_obs,end)-b.Za_theta_mean(b.ind_obs,end),2*pi,-pi)/(2*pi)));
b_unobs_mean = median(abs(mod2(b.theta_truth(b.ind_unobs,end)-b.Za_theta_mean(b.ind_unobs,end),2*pi,-pi)/(2*pi)));
a_obs_mean = median(abs(mod2(a.theta_truth(a.ind_obs,end)-a.Za_theta_mean(a.ind_obs,end),2*pi,-pi)/(2*pi)));
a_unobs_mean = median(abs(mod2(a.theta_truth(a.ind_unobs,end)-a.Za_theta_mean(a.ind_unobs,end),2*pi,-pi)/(2*pi)));

sz = 75;
lwidth = 2;
scatter(b.ind_obs,abs(mod2(b.theta_truth(b.ind_obs,end)-b.Za_theta_mean(b.ind_obs,end),2*pi,-pi)/(2*pi)),sz,'b','^','filled','LineWidth',lwidth)
hold on
scatter(b.ind_unobs,abs(mod2(b.theta_truth(b.ind_unobs,end)-b.Za_theta_mean(b.ind_unobs,end),2*pi,-pi)/(2*pi)),sz,'b','^','LineWidth',lwidth)
hold on
scatter(a.ind_obs,abs(mod2(a.theta_truth(a.ind_obs,end)-a.Za_theta_mean(a.ind_obs,end),2*pi,-pi)/(2*pi)),sz,'r','filled','LineWidth',lwidth)
hold on
scatter(a.ind_unobs,abs(mod2(a.theta_truth(a.ind_unobs,end)-a.Za_theta_mean(a.ind_unobs,end),2*pi,-pi)/(2*pi)),sz,'r','LineWidth',lwidth)
hold off
%yline([0])
yline([b_obs_mean],'b')
yline([b_unobs_mean],'--b')
yline([a_obs_mean],'r')
yline([a_unobs_mean],'--r')
set(gca,'YScale','log')
set(gca,'FontSize',18)
xlabel('$i$','Interpreter','latex','FontSize',20)
ylabel('$|\phi_i - \phi^{\rm a}_i|/(2\pi)$','Interpreter','latex','FontSize',20)
legend('standard, observed','standard, unobserved','localized, observed','localized, unobserved',...
    'Interpreter','latex','FontSize',18,'Location','southeast','NumColumns',2)
box on

%% Residuals in zeta
h = figure(14);

%scatter(1:a.N,mod2(a.phi_truth(:,end)-a.Za_phi_mean(:,end),2*pi,-pi),[],'r')
max_omega = max(abs(a.zeta_truth));

b_obs_mean = median(abs((b.zeta_truth(b.ind_obs)-b.Za_zeta_mean(b.ind_obs,end))/max_omega));
b_unobs_mean = median(abs((b.zeta_truth(b.ind_unobs)-b.Za_zeta_mean(b.ind_unobs,end))/max_omega));
a_obs_mean = median(abs((a.zeta_truth(a.ind_obs)-a.Za_zeta_mean(a.ind_obs,end))/max_omega));
a_unobs_mean = median(abs((a.zeta_truth(a.ind_unobs)-a.Za_zeta_mean(a.ind_unobs,end))/max_omega));

sz = 75;
lwidth = 2;
scatter(b.ind_obs,abs((b.zeta_truth(b.ind_obs)-b.Za_zeta_mean(b.ind_obs,end))/max_omega),sz,'b','^','filled','LineWidth',lwidth)
hold on
scatter(b.ind_unobs,abs((b.zeta_truth(b.ind_unobs)-b.Za_zeta_mean(b.ind_unobs,end))/max_omega),sz,'b','^','LineWidth',lwidth)
hold on
scatter(a.ind_obs,abs((a.zeta_truth(a.ind_obs)-a.Za_zeta_mean(a.ind_obs,end))/max_omega),sz,'r','filled','LineWidth',lwidth)
hold on
scatter(a.ind_unobs,abs((a.zeta_truth(a.ind_unobs)-a.Za_zeta_mean(a.ind_unobs,end))/max_omega),sz,'r','LineWidth',lwidth)
hold off
%yline([0])
yline([b_obs_mean],'b')
yline([b_unobs_mean],'--b')
yline([a_obs_mean],'r')
yline([a_unobs_mean],'--r')
set(gca,'YScale','log')
set(gca,'FontSize',18)
xlabel('$i$','Interpreter','latex','FontSize',20)
ylabel('$|\zeta_i - \zeta^{\rm a}_i|/\zeta_{\max}$','Interpreter','latex','FontSize',20)
legend('standard, observed','standard, unobserved','localized, observed','localized, unobserved',...
    'Interpreter','latex','FontSize',18,'Location','southeast','NumColumns',2)
box on

%% Averaged normalized covariances Pa
figure(13)
imagesc(max(log10(abs(ave_Pa)),-3))
colorbar
pbaspect([1,1,1])
set(gca,'FontSize',14)
xlabel('$\phi,\,\omega$','Interpreter','latex','FontSize',18)
ylabel('$\phi,\,\omega$','Interpreter','latex','FontSize',18)
title('log10(abs(Correlations))','FontSize',18)

