function [B,D] = dmf_aux_c(R, varargin)
    [m,~] = size(R);
    [k, rho, Y] = process_options(varargin, 'K', 64, 'rho', 0.1, 'Y', zeros(m,0));
    [P,Q] = dmf(R, 'max_iter', 20, 'rho', rho, 'alpha', 0, 'beta', 0, 'K', k);
    if isempty(Y)
        B = P; D = Q;
    else
        B = [P,P];
        D = [Q,itq(Y,k)];
    end
end


function B = itq(X, k)
%%% |X - B W'|_F^2
    debug = true;
    Xt = X';
    [~, ~, W] = svds(X, k);
    for iter=1:20
       B = proj_hamming_balance(X*W);
       W = proj_stiefel_manifold(Xt*B);
       if debug
           loss = sum(sum(X.^2)) - 2 * sum(sum((B' * X) .* W')) + sum(sum((W' * W) .* (B' * B)));
           fprintf('itq:iter=%3d, loss=%.1f\n', iter, loss)
       end
    end
end