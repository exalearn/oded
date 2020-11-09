function rxx_theta = RxxTheta(r, c, theta, blurringT)
%     [c, r] = meshgrid(c, r);
%     if (size(t1)~=size(t2))
%         error('the size must be the same')
%     end
    theta
    rxx_theta = zeros(size(r, 1), size(c, 2));
    fun = @(x, y) RyyTheta(x, y, theta)/(blurringT^2);
    for m = 1:size(r, 1)
        for n = 1:size(c, 2)
            rxx_theta(m, n) = integral2(fun, r(m)-blurringT, r(m),...
                c(n)-blurringT, c(n), 'AbsTol',1e-4);
        end
    end
end

%decomposite into r+c and r-c