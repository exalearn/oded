function ryx = Ryx(theta_cell, rho)
    ryx_theta = theta_cell{2};
    ryx = kron([1+rho^2, 2*rho; 2*rho, 1+rho^2], ryx_theta);
end

% function ryx = RYX_factor(r, c, theta, bluringT)
%     [c, r] = meshgrid(r, c);
% %     if (size(r)~=size(c))
% %         error('the size must be the same')
% %     end
%     ryx = zeros(size(r, 1), size(c, 2));
%     
%     for m = 1:size(r, 1)
%         for n = 1:size(c, 2)
%             fun = @(x) RYY_factor(r(m, n), x, theta)/bluringT;
%             ryx(m, n) = integral(fun, c(m, n)-bluringT, c(m, n));
%         end
%     end
% end
