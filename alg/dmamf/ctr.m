function [B,D] = ctr(R, varargin)
    [m,n] = size(R);
    [k, max_iter, rho, alpha, beta, D] = process_options(varargin, 'K', 64, 'max_iter', 20, 'rho', 0.1, 'alpha', 0.01, 'beta', 0.01, 'Y', zeros(n,0));
   
    fprintf('ctr(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f)\n', k, max_iter, rho);
    
    if ~isempty(D)
        if size(D,2)~=k
            [U, S, ~] = svds(D, k);
            D = U * S.^(0.5);
        end
    else
        D = zeros(n,k);
    end
    Rt = R';
    for iter=1:max_iter
        P = optimize_(Rt, P, zeros(m,k), zeros(k,m), D + Q, rho, alpha, false);
        Q = optimize_(R, Q, D, zeros(k,n), P, rho, beta, false);
        fprintf('iter=%3d, loss=%.1f\n', iter, loss_());
    end
    B = P;
    D = Q + D;
    function val = loss_()
        val = 0;
        T = Q + D;
        for u=1:m
            r = Rt(:,u);idx = r ~= 0;r = r(idx);
            r_ = T(idx, :) * P(u,:)';
            val = val + sum((r - r_).^2) - rho * sum(r_.^2);
        end
        val = val + rho * sum(sum((P'*P) .* (T'*T)));
        
    end
end

function B = optimize_(Rt, B, P, Xt, V, rho, alpha, binary)
    m = size(Rt, 2);
    VtV = V' * V;
    for i=1:m
        r = Rt(:,i); idx = r~=0; r = full(r(idx));
        Vi = V(idx,:); 
        A = (1-rho) * (Vi' * Vi) + rho * VtV;
        b = Vi' * r + alpha * Xt(:,i) - A * P(i,:)';        
        if binary
            B(i,:) = ccd_mex(B(i,:), A, b, 1);
        else
            B(i,:) = ccd_mex(B(i,:), A, b, 1, alpha);
        end
    end
end