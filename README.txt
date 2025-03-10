Matlab files used for data assimilation of coupled oscillators, including network specific localization, as discussed in "Data assimilation for networks of coupled oscillators: Inferring unknown model parameters from partial observations" by Lauren D Smith and Georg A Gottwald.

---Contents---

Main files that perform simulations, perform DA, and create figures:

KM_DA.m: DA performed with the Kuramoto model and a ring network topology (cf. Fig. 8 and Fig. 10).

KM_DA_correlations.m: Analysing the average correlations over a DA process with the Kuramoto model and a ring network topology (cf. Fig. 3).

DA_Kuramoto_ring_nobs_run.m and DA_Kuramoto_ring_nobs_run_2.m: The effect of varying the number of observed nodes for DA applied to the Kuramoto model with a ring network topology (cf. Fig. 9).

DA_Kuramoto_BA_realisations_run.m and DA_Kuramoto_ER_realisations_run.m: DA with the Kuramoto model with Barabasi-Albert and Erdos-Renyi network topologies, respectively (cf. Fig. 11).

theta_neuron_DA.m: DA performed with the a network of theta neurons and a ring-like network topology (cf. Fig. 12).

theta_neuron_DA_nobs.m and theta_neuron_DA_nobs_2.m: The effect of varying the number of observed nodes for DA applied to networks of theta neurons with a ring-like network topology (cf. Fig. 13).


Additional function that are used within the files above:

mat_exp_loc.m: Matrix exponential localisation matrix that is used throughout the main text to perform network specific localisation as part of the EnKF.

PertObs_loc_circle.m: Performs the EnKF for a state space with a combination of phase variables that live on a circle, and parameter variables that live on the real line.

PertObs_loc_circle_out_Pf.m: As above except the forcast covariance matrix Pf is also output.

correlation_func.m: The localisation function proposed by Gaspari & Cohn in "Construction of correlation functions in two and three dimensions", 1999.

ring_loc_param.m: Computes the value of lambda to use in the matrix exponential localisation function.

kuramoto_matrixform.m: ODE of the Kuramoto model, used for simulations and forecasts.

theta_matrixform.m: ODE of the theta neuron model, used for simulations and forecasts.

theta_K.m and theta_P.m: auxilliary functions needed for the theta neuron model.

mod2.m: Like regular "mod", but the output is in the range [b,b+a), where b is an input variable. Useful for doing mod 2*pi with the output in the range [-pi,pi), rather than [0,2*pi).

SFNG_modified.m: scale-free network generator (Barabasi-Albert networks). This is a slightly modified version of code that was obtained from the Matlab File Exchange:
https://au.mathworks.com/matlabcentral/fileexchange/11947-b-a-scale-free-network-generation-and-visualization





