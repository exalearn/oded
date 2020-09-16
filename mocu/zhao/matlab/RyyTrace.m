function ryy_trace = RyyTrace(r, rho, theta_cell)
    rhomat = [1+rho^2, 2*rho; 2*rho, 1+rho^2];
    ryy_trace = sum(RyyTheta(r, r, theta_cell{1}))*trace(rhomat);
end