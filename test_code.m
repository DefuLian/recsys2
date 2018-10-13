a = randn(1000,64);
a = (a>0);
b = randn(100, 64);
b = (b>0);
aa1 = compactbit(a);
bb1 = compactbit(b);
tic;c1 = mult_real(a,b);toc
tic;c2 = a*b';toc
tic;c3 = mult_binary(aa,bb);toc
norm(c1-c2)
norm(c2-c3)

%%
addpath(genpath('~/code/recsys/'))
addpath(genpath('~/code/recsys2/'))
load ~/data/yelpdata.mat
load('../dataset/yelpdata.mat')
[Traindata, Testdata] = split_matrix(data,'un', 0.8);
[m,n] = size(Traindata);
B = randn(m,8);
D = randn(n,8);
%B = 2*(B>0) - 1;
B = ones(m,8);
D = ones(n,8);
%D = 2*(D>0) - 1;
train = Traindata;
test = Testdata;
tic; topkmat = topk_finder(train, B, D, 1000, true);toc;
tic; [mat_rank, user_count, cand_count ] = predict_tuple(topkmat, test, B, D, 100); toc;
tic; eval1 = compute_item_metric(mat_rank, user_count, cand_count, 100, 100);
tic;a1 = topk_finder(train, B, D, 10, false);toc;
eval2 = evaluate_item(train, test~=0, B, D, 100, 100);

data = Traindata + Testdata;
rec_hash = @(mat) dmf(mat, 'K', 64, 'max_iter',10, 'rho',0.01, 'alpha', 0.01, 'beta',0.01);
rec_real = @(mat, varargin) gmf(mat, 'K', 64, 'max_iter',10, 'rho',0.01, 'eta', 0.01, varargin{:});
eval1 = recommender(rec_hash, rec_real, data, 'test_ratio', 0.2, 'topk', 500);
eval2 = rating_recommend(rec_real, data, 'test_ratio', 0.2);
eval3 = rating_recommend(rec_hash, data, 'test_ratio', 0.2);

A = randn(100,1000);
A = A*A' - 0.01*eye(100);
b = randn(100,1);
x = ccd_bqp_mex(zeros(100,1), A, b, 1000, 0.01);
x1 = A\b;
norm(x1-x,1)

%%
[B,D] = gmf_(train, 'K', 8, 'max_iter',10, 'rho',0.01, 'eta', 0.01);
[B1,D1] = gmf_(train, 'K', 8, 'max_iter',10, 'rho',0.01, 'eta', 0.01, 'B0', B,'D0',D);

[B2,D2] = gmf_(train, 'K', 16, 'max_iter',10, 'rho',0.01, 'eta', 0.01);
eval1 = evaluate_item(train, test, B, D, -1, 100);
eval2 = evaluate_item(train, test, [B,B1], [D,D1], -1, 100);
eval3 = evaluate_item(train, test, B2, D2, -1, 100);


%% 
load ~/result/dmf_results(new_ndcg).mat
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
clear('result');
para = paras{1}; para = cell2struct(para(2:2:end), para(1:2:end),1);
load ~/data/yelp/yelp/map/yelpdata.mat
item_feat = tfidf(item_feat);
eval1 = rating_recommend(@dmf_aux_reg, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 10);
eval2 = rating_recommend(@dmf, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'alpha', 0, 'beta',0);
eval3 = rating_recommend(@dmf_aux, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 10);
eval4 = rating_recommend(@dmf_aux, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 1);
eval5 = rating_recommend(@dmf_aux, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 0);
eval6 = rating_recommend(@dmf_aux_, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat);
eval7 = rating_recommend(@dmf_aux_, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho);
eval8 = rating_recommend(@dmf_aux_c, data, 'test_ratio', 0.2, 'K', 64, 'rho', para.rho, 'Y', item_feat);
eval9 = rating_recommend(@dmf_aux_c, data, 'test_ratio', 0.2, 'K', 64, 'rho', para.rho);
para.rho= 0.1;
metric_fun = @(metric) metric.item_recall_like(1,end);
alg1 = @(varargin) rating_recommend(@(mat) ctr(mat, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, varargin{:}), data, 'test_ratio', 0.2);
[outputs{1:4}] = hyperp_search(alg1, metric_fun, 'beta', [0.01, 0.1, 1, 10, 50, 100], 'rho',[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]);
eval = rating_recommend(@dmf, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', outputs{1}{4}, 'alpha', 0, 'beta',0, 'init',true);


eval  = rating_recommend(@ctr, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 1);
eval_1 = rating_recommend(@dmf, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'alpha', 0, 'beta',0, 'init',true);
eval_2  = rating_recommend(@ctr, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho);
eval_3  = rating_recommend(@ctr, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 10);
eval_4  = rating_recommend(@ctr, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'beta', 0.1);
eval_5 = rating_recommend(@dmf_aux_reg, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho,...
    'Y', item_feat, 'beta', 10,'init',true, 'reg', true);
eval_6 = rating_recommend(@dmf_aux_reg, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho,...
    'Y', item_feat, 'beta', 5,'init',true, 'reg', true);
eval_7 = rating_recommend(@dmf_aux_reg, data, 'test_ratio', 0.2, 'K', 64, 'max_iter', 20, 'rho', para.rho,...
    'Y', item_feat, 'beta', 1,'init',true, 'reg', true);


%[B,D] = dmf_aux_reg(data, 'K', 64, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'beta', 0.1);
load ~/data/citeulike/citeulike.mat
item_feat = tfidf(item_feat);

k=64;
[U,S,V] = svds(item_feat, k);
Y = U * S.^(0.5);

