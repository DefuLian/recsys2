function [B, D] = dmf_aux(R, varargin)
%%% sum_(i,j)\in\Omega (r_ij-1/4(p_i+b_i)'(q_j+d_j))^2 
%%% + \rho sum_(i,j)\notin\Omega(1/4(p_i+b_i)'(q_j+d_j))^2 
%%% + \alpha sum_i |W_x b_i - x_i|^2
%%% + \beta sum_j |W_y d_j - y_j|^2
%%% + \eta (sum_i |p_i|^2 + sum_j |q_j|^2)
%%% s.t. W_x^T W_x = I and W_y^T W_y = I
    [m,n] = size(R);
    [init, X, Y, opt] = process_options(varargin, 'init', false, 'X', zeros(m,0), 'Y',zeros(n,0));

    if init
        [P, Q, B, D] = dmf_aux_(R, 'init', true, 'X', X, 'Y', Y, opt{:});
    else
        [P, Q, B, D] = dmf_aux_(R, 'init', true, 'X', X, 'Y', Y, opt{:});
        %B0 = 2*(B0>0) - 1; D0 = 2*(D0>0) - 1;
        [P, Q, B, D] = dmf_aux_(R, 'init', false,'X', X, 'Y', Y, 'P0', P, 'Q0', Q, 'B0', B, 'D0', D, opt{:});
    end

    if isempty(X) && isempty(Y)
        B = P; D = Q;
    elseif ~isempty(X) && isempty(Y)
        B = [B,P];
        D = [Q,Q];
    elseif isempty(X) && ~isempty(Y)
        B = [P,P];
        D = [Q,D];
    else
        B = [B,B,P,P];
        D = [D,Q,D,Q];
    end
end
function [P, Q, B, D] = dmf_aux_(R, varargin)
    [m,n] = size(R);
    [k, max_iter, rho, alpha, beta, eta, X, Y, B, D, P, Q, init] = process_options(varargin, 'K', 64, 'max_iter', 10, ...
        'rho', 0.01, 'alpha',0, 'beta',0, 'eta', 0.01, 'X', zeros(m,0),'Y', zeros(n,0),...
        'B0', zeros(m,0), 'D0', zeros(n,0), 'P0', zeros(m,0), 'Q0', zeros(n,0), 'init', false);
    print_info();
    if ~init
        R = scale_matrix(R, k);
    end
    Rt = R';

    rng(10);
    if isempty(B)
        if ~isempty(X)
            B = randn(m,k)*0.1; 
            if ~init
                B = 2*(B>0)-1; 
            end
        else
            B = zeros(m,k);
        end
    end
    if isempty(D)
        if ~isempty(Y)
            D = randn(n,k)*0.1; 
            if ~init
                D = 2*(D>0)-1; 
            end
        else
            D = zeros(n,k);
        end
    end
    if isempty(P) 
        P = randn(m,k)*0.1; 
        if ~init
            P = 2*(P>0)-1; 
        end
    end
    if isempty(Q)
        Q = randn(n,k)*0.1;
        if ~init
            Q = 2*(Q>0)-1;
        end
    end
    
    if isempty(X) && isempty(Y)
        coef = 1;
    elseif ~isempty(X) && ~isempty(Y)
        coef = 1/4;
    else
        coef = 1/2;
    end
    
    for iter=1:max_iter
        V = coef * (Q + D);
        P = optimize_(Rt, P, B, zeros(k,m), V, rho, eta, ~init);
        if ~isempty(X)
            W_x = proj_stiefel_manifold(X'*B);
            B = optimize_(Rt, B, P, (X * W_x)', V, rho, alpha, ~init);
        end
        
        U = coef * (P + B);
        Q = optimize_(R, Q, D, zeros(k,n), U, rho, eta, ~init);
        if ~isempty(Y)
            W_y = proj_stiefel_manifold(Y'*D);
            D = optimize_(R, D, Q, (Y * W_y)', U, rho, beta, ~init);
        end
        
        loss = loss_();
        fprintf('Iteration=%3d of all optimization, loss=%.1f\n', iter, loss);
    end

function print_info()
    init_str = '';
    if init
        init_str = '_init';
    end
    if isempty(X) && isempty(Y)
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f, eta=%f)\n', init_str, k, max_iter, rho, eta);
    elseif ~isempty(X) && isempty(Y)
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f, alpha=%f, eta=%f)\n', init_str, k, max_iter, rho, alpha, eta);
    elseif isempty(X) && ~isempty(Y)
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f,  beta=%f, eta=%f)\n',init_str, k, max_iter, rho, beta, eta);
    else
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f, eta=%f)\n', init_str, k, max_iter, rho, alpha, beta, eta);
    end
end
function val = loss_()
    val = 0;
    if ~isempty(X)
        S = 1/2 * (P+B);
    else
        S = P;
    end
    if ~isempty(Y)
        T = 1/2 * (Q+D);
    else
        T = Q;
    end
    for u=1:m
        r = Rt(:,u);idx = r ~= 0;r = r(idx);
        r_ = T(idx, :) * S(u,:)';
        val = val + sum((r - r_).^2) - rho * sum(r_.^2);
    end
    val = val + rho * sum(sum((S'*S) .* (T'*T)));
    val = val + eta * norm(P,'fro')^2 + eta * norm(Q,'fro')^2;
    if ~isempty(X)
        val = val + alpha * (tracet(B, B) - 2*tracet(B, X*W_x) + tracet(X, X));
    end
    if ~isempty(Y)
        val = val + beta  * (tracet(D, D) - 2*tracet(D, Y*W_y) + tracet(Y, Y));
    end
end

end

function v = tracet(X, Y)
    %%% v = trace(X'Y);
    v = sum(sum(X.*Y));
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