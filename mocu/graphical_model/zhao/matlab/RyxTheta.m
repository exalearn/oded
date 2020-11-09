function ryx_theta = RyxTheta(r1, c1, theta, bluringT)
%     [c, r] = meshgrid(c1, r1);
%     if (size(r)~=size(c))
%         error('the size must be the same')
%     end
    ryx_theta = zeros(size(r1, 1), size(c1, 2));
    
    for n = 1:size(c1, 2)
%         for m = 1:size(r1, 1)
%         
%             fun = @(x) RyyTheta(r1(m), x, theta)/bluringT;
%             ryx(m, n) = integral(fun, c1(n)-bluringT, c1(n));
% %             fun = @(x) RyyTheta(r1(:), x, theta)/bluringT;
% %             fun = @(x) RyyTheta(r1(:, n), x, theta)/bluringT;
% %             ryx(:, n) = integral(fun, c1(n)-bluringT, c1(n), 'ArrayValued',true);
% %             ryx(:, n) = integral(fun, c1(n)-bluringT, c1(n));
%         end
        
        fun = @(x) RyyTheta(r1, x, theta)/bluringT;
        ryx_theta(:, n) = integral(fun, c1(n)-bluringT, c1(n), 'ArrayValued',true);
        
    end
end