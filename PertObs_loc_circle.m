function [Za_mean,Za] = PertObs_loc_circle(Zf,delta,H,R,RI,y_at_j,S,w,params)

ms = params.ms;
D = params.D;
Df = params.Df;
Ds = params.Ds;

%%% Mean of forecast
Zf_mean = zeros(1,D+Df);
Zf_mean(1:D) = mod( angle( mean( exp(1i*Zf(1:D,:)'))), 2*pi);
Zf_mean(D+1:D+Df) = mean(Zf(D+1:end,:)');
Zf_mean=Zf_mean';
%%% Inflation
for j=1:D
    Zf(j,:) = sqrt(delta)*mod2(Zf(j,:)-Zf_mean(j),2*pi,-pi) + Zf_mean(j);
end
for j=1:Df
    Zf(D+j,:) = sqrt(delta)*(Zf(D+j,:)-Zf_mean(D+j)) + Zf_mean(D+j);
end

    
%%% Variance of forecast Pf
A_temp = Zf - Zf_mean*ones(1,ms);
A_temp(1:D,:) = mod2(A_temp(1:D,:),2*pi,-pi);
Pf = A_temp*A_temp'/(ms-1);


%%% Localisation of variance of forecast Pf
Pf = params.locA .* Pf;


%%% Analysis
KR = Pf*H'*pinv(H*Pf*H'+R);
          
%%% Mean of analysis
Za_mean = Zf_mean - KR*(mod2(H*Zf_mean - y_at_j,2*pi,-pi));
Za_mean(1:D) = mod(Za_mean(1:D),2*pi);

% Perturbed observations to create analysis ensemble
% Scale with ensemble variance factor sqrt(ms-1) and
% Subtract the ensemble mean from perturbed observation to ensure 
% that mean(update of full Xa) = analysis mean
% See Evensen book p.165
pertobs = sqrtm(R)*randn(Ds,ms);
pertobs_mean = mean(pertobs,2);
pertobs = sqrt((ms-1)/ms)*(pertobs - pertobs_mean*ones(1,ms));

vKF = Zf - KR*( mod2(H*Zf - y_at_j*ones(1,ms) + pertobs, 2*pi,-pi) );

%%% Unconstrained
Za = vKF;
Za(1:D,:) = mod(Za(1:D,:),2*pi);

%%% Analysis mean
Za_mean = zeros(1,D+Df);
Za_mean(1:D) = mod( angle( mean( exp(1i*Za(1:D,:)'))), 2*pi);
Za_mean(D+1:D+Df) = mean(Za(D+1:end,:)');
Za_mean = Za_mean';
