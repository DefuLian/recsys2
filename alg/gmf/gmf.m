function [P, Q] = gmf(R, varargin)
%%% [B,D]=dmf(R, 'K', 64, 'max_iter', 30, 'debug', true, 'islogit', false, 'beta', 0.01, 'lambda', 0.01);
%%% optimize 
%%% \sum_{(i,j)\in \Omega} \ell(r_{ij}, b_i'd_j + p_i'q_j) + 
%%% \rho \sum_{(i,j)\notin \Omega} (b_i'd_j+p_i'q_j)^2 + 
%%% \alpha(|P|_F^2 + |Q|_F^2) + 
%%% R rating matrix of size m x n
%%% K dimension of hamming space
%%% max_iter the max number of iterations
%%% islogit is  logitloss function
%%% rho regularization coefficient for interaction/implicit regularization
%%% eta regularization coefficient
[m,n] = size(R);
[k, max_iter, debug, opt.islogit, opt.rho, B, D, opt.eta] = process_options(varargin, 'K', 64, 'max_iter', 10, 'debug', true, ...
    'islogit', false, 'rho', 0.01, 'B0',zeros(m,0), 'D0',zeros(n,0), 'eta', 0.01);
print_info();
%B = B/sqrt(2*size(B,2)); D = D/sqrt(2*size(D,2));
%if isexplit(R)
%   [I,J,V]=find(R);
%   V = (V - min(V)) ./ (max(V) - min(V)) - 0.5;
%   R = sparse(I,J,V,m,n);
%end
rng(10);
P = randn(m,k)*0.1; 
Q = randn(n,k)*0.1;
Rt = R.';
converge = false;
iter = 1;
%for iter=1:max_iter
while ~converge
    P = optimize(Rt, Q, P, D, B, opt);
    Q = optimize(R,  P, Q, B, D, opt);
    loss = loss_();
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.1f\n', iter, loss);
    end
    if iter >= max_iter
        converge = true;
    end
    iter = iter + 1;
end
function print_info()
    if opt.islogit
        fprintf('gmf_logit(K=%d, max_iter=%d, rho=%f, eta=%f)\n', k, max_iter, opt.rho, opt.eta);
    else
        fprintf('gmf(K=%d, max_iter=%d, rho=%f, eta=%f)\n', k, max_iter, opt.rho, opt.eta);
    end
end
function val = loss_()
    val = 0;
    for u=1:m
        r = Rt(:,u);
        idx = r ~= 0;
        r_ = Q(idx, :) * P(u,:)' + D(idx,:) * B(u,:)';
        r = r(idx);
        if opt.islogit
            val = val + sum(logitloss(r .* r_)) - opt.rho * sum(r_.^2);
        else
            val = val + sum((r - r_).^2) - opt.rho * sum(r_.^2);
        end
    end
    val = val + opt.rho*sum(sum((P'*P) .* (Q'*Q)));
    val = val + opt.rho*sum(sum((B'*B) .* (D'*D)));
    val = val + opt.rho*sum(sum((P'*B) .* (Q'*D))) * 2;
    val = val + opt.eta * norm(P,'fro')^2 + opt.eta * norm(Q,'fro')^2;
end
end

function P = optimize(Rt, Q, P, D, B, opt)
QtQ = Q'*Q; DtQ = D'*Q; 
max_iter = 1;
m = size(Rt, 2);
X = B*DtQ;
for u=1:m
    p = P(u,:); b = B(u,:);
    r = Rt(:,u);
    idx = r ~= 0;
    Qu = Q(idx, :); Du = D(idx,:);
    r_ = Qu * p' + Du * b';
    x = opt.rho * X(u,:) + (1 - opt.rho) * b * Du' * Qu;
    P(u,:) = ccd_logit_mex(full(r(idx)), Qu, p,  opt.rho * (QtQ - Qu'*Qu), x, r_, opt.islogit, max_iter, opt.eta);
end
end

function v = logitloss(v)
if v>-500
    v = log(1+exp(-v));
else
    v = -v;
end
end