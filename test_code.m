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
load ~/data/ml100kdata.mat
[m,n] = size(Traindata);
B = randn(m,64);
D = randn(n,64);
B = 2*(B>0) - 1;
D = 2*(D>0) - 1;
train = Traindata;
test = Testdata;
tic; topkmat = topk_finder(train, B, D, 500, true);toc;
tic; [mat_rank, user_count, cand_count ] = predict_tuple(topkmat, test, B, D, 100); toc;
tic; eval1 = compute_item_metric(mat_rank, user_count, cand_count, 100, 100);
tic;a1 = topk_finder(train, B, D, 10, false);toc;
eval2 = evaluate_item(train, test~=0, B, D, 100, 100);

data = Traindata + Testdata;
rec_hash = @(mat) dmf(mat, 'K', 64, 'max_iter',10, 'rho',0.01, 'alpha', 0.01, 'beta',0.01);
rec_real = @(mat) dmf(mat, 'K', 64, 'max_iter',10, 'rho',0.01, 'alpha', 0.01, 'beta',0.01);
eval1 = recommender(rec_hash, rec_real, data, 'test_ratio', 0.2);

eval2 = rating_recommend(rec_hash, data, 'test_ratio', 0.2);