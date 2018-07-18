function [B, D] = dmf_aux_reg(R, varargin)
%%% sum_(i,j)\in\Omega (r_ij-b_i'd_j)^2 
%%% + \rho sum_(i,j)\notin\Omega(b_i'd_j)^2 
%%% + \alpha sum_i |W_x b_i - x_i|^2
%%% + \beta sum_j |W_y d_j - y_j|^2
%%% s.t. W_x^T W_x = I and W_y^T W_y = I
    [init, alpha_init, beta_init, alpha_bin, beta_bin, opt] = process_options(varargin, 'init', false, ...
        'alpha_init',0, 'beta_init',0, 'alpha_bin',0, 'beta_bin',0);

    if init
        [B,D] = dmf_aux_r(R, 'init', true, 'alpha', alpha_init, 'beta', beta_init, opt{:});
    else
        [B,D] = dmf_aux_r(R, 'init', true, 'alpha', alpha_init, 'beta', beta_init, opt{:});
        %B0 = 2*(B0>0) - 1; D0 = 2*(D0>0) - 1;
        [B,D] = dmf_aux_r(R, 'init', false, 'alpha', alpha_bin, 'beta', beta_bin, 'B0', B, 'D0', D, opt{:});
    end

end
function [B, D] = dmf_aux_r(R, varargin)
    [m,n] = size(R);
    [k, max_iter, rho, alpha, beta, X, Y, B, D, init, reg] = process_options(varargin, 'K', 64, 'max_iter', 10, ...
        'rho', 0.01, 'alpha',0, 'beta',0,  'X', zeros(m,0),'Y', zeros(n,0),...
        'B0', zeros(m,0), 'D0', zeros(n,0), 'init', false, 'reg', false);
    
    print_info();
    if ~init
        R = scale_matrix(R, k);
        %beta = beta * k;
        %alpha = alpha * k;
    end
    Rt = R';

    rng(10);
    if isempty(B)
        B = randn(m,k)*0.1; 
        if ~init
            B = 2*(B>0)-1; 
        end
    end
    if isempty(D)
        D = randn(n,k)*0.1; 
        if ~init
            D = 2*(D>0)-1; 
        end
    end
    if ~isempty(X)
        XtX = (X'*X +speye(size(X,2),size(X,2)));
        %perm_x = symamd(XtX);
        %R_x = chol(XtX(perm_x, perm_x));
        %R_x_inv = inv(R_x);
        %clear('XtX');
    end
    if ~isempty(Y)
        YtY = (Y'*Y +speye(size(Y,2),size(Y,2)));
        %perm_y = symamd(YtY);
        %R_y = chol(YtY(perm_y, perm_y));
        %R_y_inv = inv(R_y);
        %clear('YtY');
    end
    for iter=1:max_iter
        if ~isempty(X)
            if ~reg
                W_x = proj_stiefel_manifold(X'*B);
            else
                %W_x(perm_x,:) = R_x_inv * R_x_inv' * (X(:,perm_x)' * B);
                W_x = XtX \ (X' * B);
            end
            B = optimize_(Rt, B, (X * W_x)', D, rho, alpha, ~init);
        else
            B = optimize_(Rt, B, zeros(k,m), D, rho, alpha, ~init);
        end
        
        if ~isempty(Y)
            if ~reg
                W_y = proj_stiefel_manifold(Y'*D);
            else
                %W_y(perm_y,:) =  R_y_inv * R_y_inv' * (Y(:,perm_y)' * D);
                W_y = YtY \ (Y' * D);
            end
            D = optimize_(R, D, (Y * W_y)', B, rho, beta, ~init);
        else
            D = optimize_(R, D, zeros(k,n), B, rho, beta, ~init);
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
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f)\n', init_str, k, max_iter, rho);
    elseif ~isempty(X) && isempty(Y)
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f, alpha=%f)\n', init_str, k, max_iter, rho, alpha);
    elseif isempty(X) && ~isempty(Y)
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f,  beta=%f)\n',init_str, k, max_iter, rho, beta);
    else
        fprintf('dmf_aux%s(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f)\n', init_str, k, max_iter, rho, alpha, beta);
    end
end
function val = loss_()
    val = 0;
    for u=1:m
        r = Rt(:,u);idx = r ~= 0;r = r(idx);
        r_ = D(idx, :) * B(u,:)';
        val = val + sum((r - r_).^2) - rho * sum(r_.^2);
    end
    val = val + rho * sum(sum((B'*B) .* (D'*D)));
    
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

function B = optimize_(Rt, B, Xt, D, rho, alpha, binary)
    m = size(Rt, 2);
    DtD = D' * D;
    for i=1:m
        r = Rt(:,i); idx = r~=0; r = full(r(idx));
        Di = D(idx,:); 
        A = (1-rho) * (Di' * Di) + rho * DtD;
        b = Di' * r + alpha * Xt(:,i);        
        if binary
            B(i,:) = ccd_mex(B(i,:), A, b, 1);
        else
            B(i,:) = ccd_mex(B(i,:), A, b, 1, alpha+1e-3);
        end
    end
end