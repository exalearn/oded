function [emocu_rho, emocu_theta] = EMocuRhoTheta(theta_dispara, rho_dispara, r, c, blurringT, sigma_)
%Author: Guang Zhao
%Date:   Dec. 2018
%emocu_rho  = E_Theta{E_Rho[MSE_rho(filter_IBR_rho(theta))]} the first term of MOCU
%emocu_theta = E_Rho{E_Theta[MSE_theta(filter_IBR_theta(rho))]} 
%theta_dispara is parameter for the distribution of theta, the width of uniform
%distribution
%rho_dispara is parameter for the distribution of rho, 2*beta(Rho, Rho)-1


%Monta Carlo
N = 400;
rho_samples = 2*betarnd(rho_dispara, rho_dispara, 1, N)-1;
theta_samples = rand(1, N)*theta_dispara-0.5*theta_dispara+5;% theta is uniform distribution or Gaussian distribution or...
ocu_rho_array = zeros(N, 1);
ocu_theta_array = zeros(N, 1);
filename = sprintf('Thetaf%.1fRho%.1f.txt', theta_dispara, rho_dispara);
fileID = fopen(filename, 'a');
for n = 1:N
    
    n
%     %calculate covariance factor related to theta
%     ryx_theta = RyxTheta(r, c, theta_samples(n), blurringT);
%     rxx_theta = RxxTheta(r, c, theta_samples(n), blurringT);
%     theta_cell = cell(3, 1);
%     theta_cell{1} = theta_samples(n);
%     theta_cell{2} = ryx_theta;
%     theta_cell{3} = rxx_theta;
%     
%     %IBR filter over rho
%     IBR_filter_rho = IbrFilterRho(theta_cell, Rho, sigma_);
    
    %calculate OCU over rho
    tic
    [ocu_rho_array(n), ocu_theta_array(n)] = OcuRhoTheta(theta_samples(n), rho_samples(n),...
        rho_dispara, theta_dispara,  r, c, blurringT, sigma_);
    
    fprintf(fileID, '%12f %12f\n', ocu_rho_array(n), ocu_theta_array(n));
    
    toc
    
end
fclose(fileID);
% rhopdf = @(rho) betapdf(rho, rho_dispara, rho_dispara);
% thetapdf = 1/theta_dispara;
% fun = @(theta, rho) OcuRho(theta, rho, rho_dispara, theta_dispara, r, c, blurringT, sigma_).*rhopdf(y)*thetapdf;
% emocu = integral2(fun, 1-theta_dispara/2, 1+theta_dispara/2,...
%             -1, 1, 'AbsTol',1e-1);

emocu_rho = mean(ocu_rho_array);
emocu_theta = mean(ocu_theta_array);
end

function [ocu_rho, ocu_theta] = OcuRhoTheta(theta, rho, rho_dispara, theta_dispara, r, c, blurringT, sigma_)
%ocu for a given theta and rho
%ocu_rho  = MSE(filter_IBR_rho(theta)) only the first term of OCU
%ocu_theta = MSE(filter_IBR_theta(rho)) 
%rho_dispara, theta_dispara is the parameter for distribution of rho and
%theta. 
%rho is beta distribution over [-1, 1]: 2*betarnd(rho_dispara, rho_dispara)-1
%theta is uniform distribution over [1-theta_para/2, 1+theta_para/2]: rand(1, N)*theta_dispara-0.5*theta_dispara+1;
%blurringT is rec blurring filter length
%sigma_ is noise parameter

%calculate covariance factor related to theta
ryx_theta = RyxTheta(r, c, theta, blurringT);
rxx_theta = RxxTheta(r, c, theta, blurringT);

%define a theta cell for computation speed up
theta_cell = cell(3, 1);
theta_cell{1} = theta;
theta_cell{2} = ryx_theta;
theta_cell{3} = rxx_theta;

%IBR filter over rho
IBR_filter_rho = IbrFilterRho(theta_cell, rho_dispara, sigma_);
IBR_filter_theta = IbrFilterTheta(rho, theta_dispara, r, c, blurringT, sigma_);

%convariance matrix
ryy_trace = RyyTrace(r, rho, theta_cell);
ryx = Ryx(theta_cell, rho);
rxx = Rxx(theta_cell, rho, sigma_);

%IBR filter over rho
ocu_rho = MseOfFilter(ryy_trace, ryx, rxx, IBR_filter_rho);
ocu_theta = MseOfFilter(ryy_trace, ryx, rxx, IBR_filter_theta);

end