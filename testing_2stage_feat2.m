parpool('local', 10);
addpath(genpath('~/code/recsys/'))
addpath(genpath('~/code/recsys2/'))
dir = '~/data';

load ~/data/amazon/map/amazondata.mat
item_feat = tfidf(item_feat);


file_name = sprintf('%s/two_stage_feat_amazon_results.mat',dir);
if exist(file_name, 'file')
    load(file_name);
end
if ~exist('result', 'var')
    result = cell(7,2);
end

if isempty(result{7,1})
    alg1 = @(varargin) rating_recommend(@(mat) ctr(mat, 'K', 64, 'max_iter', 20, 'Y', item_feat, varargin{:}), data, 'test_ratio', 0.2);
    [outputs{1:4}] = hyperp_search(alg1, @(metric) metric.item_recall_like(1,end), 'beta', [0.01, 0.1, 1, 10, 50, 100], 'rho',[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]);
    result{7,1} = outputs;
    save(file_name, 'result');
end
exit();
para = result{7,1}{1}; para = cell2struct(para(2:2:end), para(1:2:end),2);
rec_hash1 = @(mat) dmf_aux_(mat, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat);
rec_hash2 = @(mat) dmf(mat, 'K', 64, 'max_iter',20, 'rho', para.rho, 'alpha', 0, 'beta', 0);
rec_real = @(mat) ctr(mat, 'K', 64, 'max_iter',20, 'Y', item_feat, 'rho', para.rho, 'beta', para.beta);

[outputs{1:3}] = rating_recommend(rec_real, data, 'test_ratio', 0.2, 'topk', 200);
result{7,2} = outputs;

topks = [500,1000,1500,2000,2500,3000];
for tk = 1:length(topks)
    [outputs{1:3}] = recommender(rec_hash1, rec_real, data, 'test_ratio', 0.2, 'topk', topks(tk), 'cutoff', 200);
    result{tk,1} = outputs;
    [outputs{1:3}] = recommender(rec_hash2, rec_real, data, 'test_ratio', 0.2, 'topk', topks(tk), 'cutoff', 200);
    result{tk,2} = outputs;
    save(file_name, 'result');
end



