function IBR_filter_rho = IbrFilterRho(theta_cell, beta_, sigma_)
    % given a theta value, covariance is kron product of theta part and rho
    %part, here the IBR is over rho, which is beta distributed
    %theta cell is a temporary setting to avoid repeat calculation
    ryx_theta = theta_cell{2};
    rxx_theta = theta_cell{3};
    betamat = [1+1/(2*beta_+1), 0; 0, 1+1/(2*beta_+1)];
    ryx_IBR_rho = kron(betamat, ryx_theta);
    rxx_IBR_rho = kron(betamat, rxx_theta)...
    +sigma_^2*eye(2*size(ryx_theta));
    IBR_filter_rho = ryx_IBR_rho/rxx_IBR_rho;
end