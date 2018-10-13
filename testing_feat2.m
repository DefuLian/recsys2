parpool('local', 5);
addpath(genpath('~/code/recsys'))
addpath(genpath('~/code/recsys2'))

load ~/result/dmf_results(new_ndcg).mat
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
clear('result');
para = paras{2}; para = cell2struct(para(2:2:end), para(1:2:end),1);
load ~/data/amazon/map/amazondata.mat
item_feat = tfidf(item_feat);
metric_fun = @(metric) metric.item_ndcg_score(1,end);

filename = '~/data/testing_feat_large_results_amazon.mat';
if exist(filename, 'file')
    load(filename);
end

k = 64;
alg{1} = @(mat) dmf(mat, 'K', k, 'max_iter', 20, 'rho', para.rho, 'alpha',0, 'beta', 0);
alg{2} = @(mat) dmf_aux_(mat, 'K', k, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat);
alg{3} = @(mat) dmf_aux_c(mat, 'K', k, 'rho', para.rho, 'Y', item_feat);

if ~exist('outputs1','var')
    alg1 = @(mat,varargin) dmf_aux_reg(mat, 'reg', false, 'K', k, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'init', true, varargin{:});
    [outputs1{1:4}] = hyperp_search(@(varargin) rating_recommend(alg1, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'beta_init', [0,1,10,50,100,500,1000]);
    alg1 = @(mat,varargin) dmf_aux_reg(mat, 'reg', false, 'K', k, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'beta_init', outputs1{1}{2}, varargin{:});
    [outputs2{1:4}] = hyperp_search(@(varargin) rating_recommend(alg1, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'beta_bin', [0,1,10,50,100,500,1000]);
    save(filename, 'outputs1', 'outputs2');    
end
alg{4} = @(mat) dmf_aux_reg(mat, 'reg', false, 'K', k, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'beta_init', outputs1{1}{2}, 'beta_bin', outputs2{1}{2});

if ~exist('outputs3','var')
    alg1 = @(mat,varargin) dmf_aux_reg(mat, 'reg', true, 'K', k, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'init', true, varargin{:});
    [outputs3{1:4}] = hyperp_search(@(varargin) rating_recommend(alg1, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'beta_init', [0,1,10,50,100,500,1000]);
    alg1 = @(mat,varargin) dmf_aux_reg(mat, 'reg', true, 'K', k, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'beta_init', outputs3{1}{2}, varargin{:});
    [outputs4{1:4}] = hyperp_search(@(varargin) rating_recommend(alg1, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'beta_bin', [0,1,10,50,100,500,1000]);
    save(filename, 'outputs3', 'outputs4', '-append');
end
alg{5} = @(mat) dmf_aux_reg(mat, 'reg', true, 'K', k, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'beta_init', outputs3{1}{2}, 'beta_bin', outputs4{1}{2});

alg{6} = @(mat) dmf_aux_(mat, 'K', k, 'max_iter', 20, 'rho', para.rho, 'Y', item_feat, 'update', false);
alg{7} = @(mat) dmf(mat, 'K', k, 'max_iter', 20, 'rho', para.rho, 'alpha',0, 'beta', 0);

if ~exist('results', 'var')
    results = cell(length(alg),1);
end

for i=1:length(alg)
    if i>length(results) || isempty(results{i})
        fprintf('%d\n',i);
        [outputs_t{1:3}] = rating_recommend(alg{i}, data, 'test_ratio', 0.2, 'times', 5);
        results{i} = outputs_t;
        save(filename, 'results','-append');
    end
end
