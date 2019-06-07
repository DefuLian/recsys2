addpath(genpath('~/code/recsys'))
addpath(genpath('~/code/recsys2'))

load ~/result/tkde_dcf/dmf_results(new_ndcg).mat
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
clear('result');
para = paras{2}; para = cell2struct(para(2:2:end), para(1:2:end),1);
load ~/data/amazon/map/amazondata.mat
item_feat = tfidf(item_feat);

load('~/result/tkde_dcf/testing_feat_large_results_amazon.mat')

[train,test] = split_matrix(data, 'un', 0.8);
k = 64;
[B1,D1] = dmf_aux_(train, 'K', k, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat);
[B2,D2] = dmf_aux_(train, 'K', k, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'update', false);
[B3,D3] = dmf_aux_c(train, 'K', k, 'rho', para.rho, 'Y', item_feat);

save('~/result/tkde_dcf/data.mat', 'B1','B2','B3','D1','D2','D3');


