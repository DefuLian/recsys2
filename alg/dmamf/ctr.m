function [B,D] = ctr(R, varargin)
    [m,n] = size(R);
    [k, max_iter, rho, alpha, beta, Y] = process_options(varargin, 'K', 64, 'max_iter', 20, 'rho', 0.1, 'alpha', 0.01, 'beta', 0.01, 'Y', zeros(n,0));
   
    fprintf('ctr(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f)\n', k, max_iter, rho, alpha, beta);
    if ~isempty(Y)
        [U, S, W] = svds(Y, k);
        D = NormalizeFea(U * S);
    else
        D = zeros(n,k);
    end
    Rt = R';
    P = randn(m,k)*0.1;
    Q = randn(n,k)*0.1;
    for iter=1:max_iter
        P = optimize_(Rt, P, zeros(m,k), zeros(k,m), Q, rho, alpha, false);
        Q = optimize_(R,  Q, zeros(n,k), D', P, rho, beta, false);
        %if ~isempty(Y)
        %    W = proj_stiefel_manifold(Y'*D);
        %    D = (Y * W + beta * Q) / (1+ beta);
        %end
        fprintf('iter=%3d, loss=%.1f\n', iter, loss_());
    end
    B = P;
    D = Q;
    function val = loss_()
        val = 0;
        T = Q;
        for u=1:m
            r = Rt(:,u);idx = r ~= 0;r = r(idx);
            r_ = T(idx, :) * P(u,:)';
            val = val + sum((r - r_).^2) - rho * sum(r_.^2);
        end
        val = val + rho * sum(sum((P'*P) .* (T'*T)));
        val = val + beta * norm(Q-D,'fro')^2 + alpha * norm(P,'fro')^2;
        %if ~isempty(Y)
        %    val = val + sum(sum(Y.^2)) - 2 * sum(sum((D' * Y) .* W')) + sum(sum((W' * W) .* (D' * D)));
        %end
    end
end

function B = optimize_(Rt, B, P, Xt, V, rho, alpha, binary)
    m = size(Rt, 2);
    k = size(B,2);
    VtV = V' * V;
    parfor i=1:m
        r = Rt(:,i); idx = r~=0; r = full(r(idx));
        Vi = V(idx,:); 
        A = (1-rho) * (Vi' * Vi) + rho * VtV;
        b = Vi' * r + alpha * Xt(:,i) - A * P(i,:)';
        B(i,:) = (A + alpha*diag(ones(k,1)))\b;
        %if binary
        %    B(i,:) = ccd_mex(B(i,:), A, b, 1);
        %else
        %    B(i,:) = ccd_mex(B(i,:), A, b, 1, alpha);
        %end
    end
end