function [B, D] = gmf_(R, varargin)
%%% sum_(i,j)\in\Omega (r_ij - p_i'q_j - b_i'd_j)^2 
%%% + \rho sum_(i,j)\notin\Omega(p_i'q_j + b_i'd_j)^2 
%%% + \alpha sum_i |b_i - x_i|^2
%%% + \beta sum_j |d_j - y_j|^2
%%% R rating matrix of size m x n
%%% K dimension of hamming space
%%% max_iter the max number of iterations
%%% islogit is  logitloss function
%%% rho regularization coefficient for interaction/implicit regularization
%%% eta regularization coefficient
[m,n] = size(R);
[k, max_iter, debug, opt.islogit, opt.rho, P, Q, opt.eta] = process_options(varargin, 'K', 64, 'max_iter', 10, 'debug', true, ...
    'islogit', false, 'rho', 0.01, 'B0',zeros(m,0), 'D0',zeros(n,0), 'eta', 0.01);
print_info();
R = scale_matrix(R, k+size(P,2));
%B = B/sqrt(2*size(B,2)); D = D/sqrt(2*size(D,2));
%if isexplit(R)
%   [I,J,V]=find(R);
%   V = (V - min(V)) ./ (max(V) - min(V)) - 0.5;
%   R = sparse(I,J,V,m,n);
%end
rng(10);
X = zeros(m,k)';
Y = zeros(n,k)';
B = randn(m,k)*0.1; 
D = randn(n,k)*0.1;
Rt = R.';
converge = false;
iter = 1;
%for iter=1:max_iter
while ~converge
    B = optimize_(Rt, B, P, X, D, Q, opt.rho, opt.eta, true);
    D = optimize_(R,  D, Q, Y, B, P, opt.rho, opt.eta, true);
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
        r_ = D(idx, :) * B(u,:)' + Q(idx,:) * P(u,:)';
        r = r(idx);
        if opt.islogit
            val = val + sum(logitloss(r .* r_)) - opt.rho * sum(r_.^2);
        else
            val = val + sum((r - r_).^2) - opt.rho * sum(r_.^2);
        end
    end
    val = val + opt.rho * sum(sum((B'*B) .* (D'*D)));
    val = val + opt.rho * sum(sum((P'*P) .* (Q'*Q)));
    val = val + opt.rho * sum(sum((B'*P) .* (D'*Q))) * 2;
    val = val + opt.eta * norm(B - X','fro')^2 + opt.eta * norm(D - Y','fro')^2;
end
end

function B = optimize_(Rt, B, P, X, D, Q, rho, alpha, binary)
    m = size(Rt, 2);
    Dt = D';
    DtD = Dt * D;
    U = Dt * Q * P';
    for i=1:m
        r = Rt(:,i); idx = r~=0; r = full(r(idx));
        p = P(i,:)';
        Du = Dt(:,idx); Qu = Q(idx,:);
        A = (1-rho) * (Du * Du') + rho * DtD;
        b = Du * r + alpha * X(:,i) - ((1 - rho) * Du * (Qu * p) + rho * U(:,i));
        if binary
            B(i,:) = ccd_mex(B(i,:), A, b, 1);
        else
            B(i,:) = ccd_mex(B(i,:), A, b, 1, alpha);
        end
    end
end

function B = optimize(Rt, B, P, X, D, Q, rho, alpha, binary)
    m = size(Rt, 2);
    Dt = D';
    DtD = Dt * D; 
    U = Dt * Q * P';
    for i=1:m
        r = Rt(:,i); idx = r ~= 0; r = full(r(idx));
        p = P(i,:)'; b = B(i,:)';
        Du = D(idx, :); Qu = Q(idx, :);
        r_ = Du * b + Qu * p;
        x = rho * U(:,i) - rho * Du' * Qu * p;
        if binary
            B(i,:) = ccd_logit_mex(r, Du, b,  rho * (DtD - Du'*Du), -x + alpha*X(:,i), r_, false, 1);
        else
            B(i,:) = ccd_logit_mex(r, Du, b,  rho * (DtD - Du'*Du), -x + alpha*X(:,i), r_, false, 1, alpha);
        end
    end
end

function v = logitloss(v)
if v>-500
    v = log(1+exp(-v));
else
    v = -v;
end
end
