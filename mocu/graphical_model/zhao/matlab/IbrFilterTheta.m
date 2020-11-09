function IBR_filter_theta = IbrFilterTheta(rho, theta_dispara, r, c, blurringT, sigma_)
    %theta is distributed among (1-Theta/2, 1+Theta/2) with pdf 1/Theta. 
    theta_center = 5;
    
    %theta distribution
    upperbound =  theta_center+theta_dispara/2;
    lowerbound =  theta_center-theta_dispara/2;
    thetapdf = @(theta) 1/theta_dispara;

%     N = 5000;
    fun_yx = @(theta) RyxTheta(r, c, theta, blurringT)*thetapdf(theta);
    fun_xx = @(theta) RxxTheta(r, c, theta, blurringT)*thetapdf(theta);
%     fun3 = @(theta) RyxTheta(51, 43, theta, blurringT)./theta_dispara;
%     integral(fun3, lowerbound, upperbound,  'ArrayValued',true);
%     ryx_theta_dis = integral(fun_yx, lowerbound, upperbound);
    ryx_theta_dis = integral(fun_yx, lowerbound, upperbound,  'ArrayValued',true, 'AbsTol',1e-3);
    rxx_theta_dis = integral(fun_xx, lowerbound, upperbound,  'ArrayValued',true, 'AbsTol',1e-3);
    
    rhomat = [1+rho^2, 2*rho; 2*rho, 1+rho^2];
    
    ryx_IBR_theta = kron(rhomat, ryx_theta_dis);
    rxx_IBR_theta = kron(rhomat, rxx_theta_dis)...
    +sigma_^2*eye(2*size(rxx_theta_dis));
    IBR_filter_theta = ryx_IBR_theta/rxx_IBR_theta;
end