clear
% rng default
% addpath('./corralation-matrix-calculate/');

%%
N = 99;
rho  = 0.8;
theta_dispara = 0.5;
rho_dispara = 0.5;
blurringT  = 10;
sigma_ = sqrt(0.01);
c = 0:3:N;%column number correspond to the column of matrix
r = (0:3:N)';% the row number correspond to the row of matrix
% OcuRho(r, c, rho, theta, beta_, blurringT, sigma_)
% mocu_rho = MocuRho(theta, beta_, r, c, blurringT, sigma_)
% OcuRho2(1, rho, rho_dispara, r, c, blurringT, sigma_)

% IBR_filter_theta = IbrFilterTheta(rho, theta_dispara, r, c, blurringT, sigma_)
% tic
EMocuRhoTheta(theta_dispara, rho_dispara, r, c, blurringT, sigma_)
% toc
rho_dispara = [0.5, 1.5, 5];% variance is 1/(1+2*rhodispara) larger means variance is smaller
theta_dispara = [0.5, 1.5,  2];% variance is theta_dispara^2/12
mocurho = zeros(3);
mocutheta = zeros(3);
fileID = fopen('data.txt', 'a');
fprintf(fileID, '%12s, %12s, %12s, %12s\n', 'thetapara', 'rhopara', 'mocurho', 'mocutheta');
for m = 1:3
    for n = 1:3
        [mocurho(m, n), mocutheta(m, n)] = EMocuRho(theta_dispara(m), rho_dispara(n), r, c, blurringT, sigma_);
        fprintf(fileID, '%12f, %12f, %12f, %12f\n', theta_dispara(m), rho_dispara(n),mocurho(m, n), mocutheta(m, n));
        n
    end
end

fclose(fileID);

save mocu.mat