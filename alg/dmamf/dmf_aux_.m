function [B,D] = dmf_aux_(R, varargin)
    [m,n] = size(R);
    [k, max_iter, rho, Y, P, Q, update] = process_options(varargin, 'K', 64, 'max_iter', 20, 'rho', 0.1, 'Y', zeros(m,0), 'P', [], 'Q', [], 'update', true);
    if isempty(P) || isempty(Q)
        [P,Q] = dmf(R, 'init', true, 'max_iter', 20, 'rho', rho, 'alpha', 0, 'beta', 0, 'K', k);
    end
    
    fprintf('dmf_auc(K=%d, max_iter=%d, rho=%f)\n', k, max_iter, rho);
    
    if ~isempty(Y)
        D = itq(Y, k);
        coef = 1/2;
    else
        D = zeros(n,k);
        coef = 1;
    end
    R = scale_matrix(R, k);
    Rt = R';
    P = optimize_(Rt, P, zeros(m,k), zeros(k,m), Q, rho, 0, true);
    Yt = Y';
    for iter=1:max_iter
        U = coef * P;
        Q = optimize_(R, Q, D, zeros(k,n), U, rho, 0, true);
        if ~isempty(Y) && update
            W = proj_stiefel_manifold(Yt*[Q,D]);
            W = W(:,(k+1):(2*k));
            D = proj_hamming_balance(Y*W);
        end
        [l1,l2] = loss_();
        fprintf('iter=%3d, loss1=%.1f, loss2=%.2f\n', iter, l1, l2);
        
        V = coef * ( D + Q);
        P = optimize_(Rt, P, zeros(m,k), zeros(k,m), V, rho, 0, true);
    end
    if isempty(Y)
        B = P; D = Q;
    else
        B = [P,P]; D = [Q,D];
    end
    function [l1, l2] = loss_()
        val = 0;
        if ~isempty(Y)
            T = 1/2 * (Q+D);
        else
            T = Q;
        end
        for u=1:m
            r = Rt(:,u);idx = r ~= 0;r = r(idx);
            r_ = T(idx, :) * P(u,:)';
            val = val + sum((r - r_).^2) - rho * sum(r_.^2);
        end
        l1 = val + rho * sum(sum((P'*P) .* (T'*T)));
        if isempty(Y) || ~update
            l2 = 0;
        else
            l2 = sum(sum(Y.^2)) - 2 * sum(sum((D' * Y) .* W')) + sum(sum((W' * W) .* (D' * D)));
        end
    end
end

function B = itq(X, k)
%%% |X - B W'|_F^2
    %debug = true;
    %Xt = X';
    [U, ~, ~] = svds(X, k);
    %B = +(U>0) * 2 -1;
    B = proj_hamming_balance(U);
    %for iter=1:1
    %    B = proj_hamming_balance(X*W);
    %    W = proj_stiefel_manifold(Xt*B);
    %    if debug
    %        loss = sum(sum(X.^2)) - 2 * sum(sum((B' * X) .* W')) + sum(sum((W' * W) .* (B' * B)));
    %        fprintf('itq:iter=%3d, loss=%.1f\n', iter, loss)
    %    end
    %end
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