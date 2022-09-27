%BPATH1  Brownian path simulation
close all
clear all
clc

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
set(0,'defaulttextinterpreter','tex')
set(0, 'defaultAxesTickLabelInterpreter','tex'); 
set(0, 'defaultLegendInterpreter','tex');

% % Load Filename
directory = pwd;
filename2 = [directory, '/Results_jmp_ez_util_nopol_phi0pt005_v4'];
load( filename2);

% Save Filename
filename3 = [filename2,'_Sims_update'];

varphi_t = 0.00173;

% % Initial Values
T_0 = 1.0;
R_0 = 500.0;
K_0 = 195000;

% % time discretization
T = 40; 
pers = 12*T; 
dt = T/pers;

% % Set seed for random number generator
% randn('state',100)           % set the state of randn
% rand('state',105);          % set the state of randn

% % Create control functions from optimal solutions
Nfunc = griddedInterpolant(r_mat,t_mat,N,'spline');
irfunc = griddedInterpolant(r_mat,t_mat,ir,'spline');

N_func = @(x) Nfunc(x(:,1),x(:,2));
ir_func = @(x) max(irfunc(x(:,1),x(:,2)),0);


% % Create function handle for the drifts
muT = @(x) varphi_t.*N_func(x(:,1:2));
muR = @(x) -N_func(x(:,1:2))+Gamma.*x(:,1).*(ir_func(x(:,1:2)).^thet);
muK = @(x) x(:,3).*(log(B)-delta2.*log(x(:,3))...
    +delta1.*log(C1.*exp(-eta.*x(:,2)).*AC.*(x(:,3).^gamma).*(L1.^alpha)...
    .*((N_func(x(:,1:2))-ir_func(x(:,1:2)).*x(:,1)).^(nu.*(1-alpha-gamma))).*((G.*(1-L1).^(beta)).^((1-nu).*(1-alpha-gamma)))));

% % Drift functions
drifts = @(x) [muR(x),muT(x),muK(x)];

% % Create function handles for the vols
sigmaT = @(x) sigma_t;
sigmaR = @(x) x(:,2).*sigma_r;
sigmaK = @(x) x(:,3).*sigma_k;

% % vlatility functions
vols = @(x) [sigmaR(x),sigmaT(x),sigmaK(x)];

%%%Create function handles for the arrival rate
lam_func = griddedInterpolant(r_mat,t_mat,lambda22,'spline');

%%%set bounds
k_min = 0;
k_max = 1000000;

nk = 100;
k = linspace(k_min,k_max,nk)';
hk = k(2) - k(1);

upperBounds = [max(r),max(t),max(k)];
lowerBounds = [min(r),min(t),min(k)];

% state vector dimensions
nDims = 3;

% Numer of iterations for the simulation
% With shocks on, its_par
% With shocks off, its
its_par = 10000;
its = 1;

%%%set up cell to store simulation data
hists = zeros(pers,nDims,its); 
hists2 = hists;
N_hists = zeros(pers,its);
N_hists2 = N_hists;
Price_C_iters=N_hists;
Price_R_iters=N_hists;
Price_G_iters=N_hists;
POtilde_iters=N_hists;

tic

% % Turn on parfor for parallel computing
parfor iters = 1:its_par
% for iters = 1:1
    
%%%Preallocate for history
hist = zeros(pers,nDims);
hist2 = zeros(pers,nDims);
N_hist = zeros(pers,1);
N_hist2 = zeros(pers,1);
ir_hist = zeros(pers,1);
ir_hist2 = zeros(pers,1);
jump_realization = zeros(pers,1);

%%%Fill in first point
hist(1,:) = [R_0,T_0,K_0];

hist2(1,:) = hist(1,:);

N_hist(1,:) = N_func(hist(1,1:2)); 
N_hist2(1,:) = N_func(hist2(1,1:2)); 
ir_hist(1,:) = ir_func(hist(1,1:2)); 
ir_hist2(1,:) = ir_func(hist2(1,1:2)); 

for j = 2:pers

% % turn on normrnd for brownian shocks    
% shock = normrnd(0,sqrt(dt), 1, nDims);
shock = zeros(1, nDims);

% vector calculation of states
% hist(j,:) = max(min(hist(j-1,:) + drifts(hist(j-1,:)) * dt + (vols(hist(j-1,:)).* shock), upperBounds), lowerBounds);

% individual calculation of states
hist2(j,1) = max(min(hist2(j-1,1).*exp((muR(hist2(j-1,:))./hist2(j-1,1)-0.5.*(sigmaR(hist2(j-1,:))./hist2(j-1,1)).^2 )* dt ...
                                  +sigmaR(hist2(j-1,:))./hist2(j-1,1).* shock(:,1)), upperBounds(:,1)), lowerBounds(:,1));
hist2(j,2) = max(min(hist2(j-1,2) + muT(hist2(j-1,:)) * dt + sigmaT(hist2(j-1,:)).* shock(:,2), upperBounds(:,2)), lowerBounds(:,2));
hist2(j,3) = max(min(hist2(j-1,3).*exp((muK(hist2(j-1,:))./hist2(j-1,3)-0.5.*(sigmaK(hist2(j-1,:))./hist2(j-1,3)).^2 )* dt ...
                                  +sigmaK(hist2(j-1,:))./hist2(j-1,3).* shock(:,3)), upperBounds(:,3)), lowerBounds(:,3));                              

% % updated control values
N_hist(j) =  N_func(hist(j-1,1:2));
N_hist2(j) =  N_func(hist2(j-1,1:2));
ir_hist(j) =  ir_func(hist(j-1,1:2));
ir_hist2(j) =  ir_func(hist2(j-1,1:2));

% % calculating jump realization
DU = rand(1); 
ldt = lam_func(hist(j-1,1:2))*dt; % one jump prob.
ul = (1-ldt)/2; 
ur = (1+ldt)/2; % Set centered jump

% probability thresholds,
jump_realization(j) = (DU <= ur && DU >= ul); % Get jump if prob. in [ul,ur]:

end

hists(:,:,iters) = hist;
hists2(:,:,iters) = hist2;
N_hists(:,iters) = N_hist;
N_hists2(:,iters) = N_hist2;
ir_hists(:,iters) = ir_hist;
ir_hists2(:,iters) = ir_hist2;
   
end
toc

% % create vectors for simulated outcomes
Rp_iters = squeeze(hists2(:,1,:))';
Tp_iters = squeeze(hists2(:,2,:))';
Kp_iters = squeeze(hists2(:,3,:))';
Np_iters = squeeze(N_hists2)';
irp_iters = squeeze(ir_hists2)';

toc;

% % save the simulated results
% stop
save(filename3);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % Plot the outcomes
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure
plot((Rp_iters))
title('Fossil Fuel Stock')
ylabel('R_t')
xlabel('Time')
% axis([0,200,0,2])
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
print('reserves_avg_extrir','-dpng')

figure
plot((Tp_iters))
title('Temperature')
ylabel('T_t')
xlabel('Time')
% axis([0,200,0,2])
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
print('temp_avg_extrir','-dpng')

figure
plot((Kp_iters))
title('Capital')
ylabel('K_t')
xlabel('Time')
% axis([0,200,0,2])
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
print('temp_avg_extrir','-dpng')

figure
plot((Np_iters))
title('Fossil Fuel Extraction')
ylabel('N_t')
xlabel('Time')
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')

figure
plot((Rp_iters.*irp_iters))
title('Fossil Fuel Exploration')
ylabel('I_{R,t}')
xlabel('Time')
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')

figure
plot((Np_iters-irp_iters.*Rp_iters))
title('Fossil Fuel Extraction')
ylabel('O_t')
xlabel('Time')
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')


lambda_t = exp(-cumsum(theta.*(1-exp(-phi.*(Tp_iters'.^4)))));

prob_t = (lambda_t');

fig = figure; 
plot(prob_t,'-k','LineWidth',3)
title('Cum. Probability')
ylabel('Probability')
xlabel('Time')
set(findall(gcf,'type','axes'),'fontsize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
