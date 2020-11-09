function ryy = RyyTheta(r, c, theta)
%return a matrix of size(t1) = size(t2)
%the autocorrelation is calculated by RYY = kron([1+rho^2, 2*rho; 2*rho, 1+rho^2], ryy);
%     [c, r] = meshgrid(c, r);
    ryy = 1./(2*theta).*(exp(0.01*theta*(r+c))-exp(0.01*theta*abs(r-c)));
    ryy = ryy.*(r>0).*(c>0);
end