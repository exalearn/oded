function rxx = Rxx(theta_cell, rho, sigma)
    rxx_theta = theta_cell{3};
    rxx = kron([1+rho^2, 2*rho; 2*rho, 1+rho^2], rxx_theta)...
    +sigma^2*eye(2*size(rxx_theta));
end