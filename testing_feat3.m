parpool('local', 10);
addpath(genpath('~/code/recsys'))
addpath(genpath('~/code/recsys2'))


load ~/data/citeulike/citeulike.mat
item_feat = tfidf(item_feat);
metric_fun = @(metric) metric.item_ndcg(1,end);

k = 64;
rho_array = 0.01:0.01:0.1;
alg1 = @(mat,varargin) dmf(mat, 'K', k, 'max_iter', 20, 'alpha',0, 'beta', 0, varargin{:});
rho = hyperp_search(@(varargin) item_recommend(alg1, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'rho', rho_array);
para.(rho{1}) = rho{2};
display(para)
alg1 = @(mat,varargin) dmf_aux_(mat, 'K', k, 'max_iter', 20, 'Y', item_feat, varargin{:});
rho = hyperp_search(@(varargin) item_recommend(alg1, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'rho', rho_array);
display(rho)
filename = '~/data/testing_feat_citeulike_results2.mat';
if exist(filename, 'file')
    load(filename);
end


alg{1} = @(mat) dmf(mat, 'K', k, 'max_iter', 20, 'rho', para.rho, 'alpha',0, 'beta', 0);
alg{2} = @(mat) dmf_aux_(mat, 'K', k, 'max_iter', 20, 'rho', rho{2}, 'Y', item_feat);
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


if ~exist('results', 'var')
    results = cell(length(alg),1);
end

for i=1:length(alg)
    if isempty(results{i})
        [outputs_t{1:3}] = rating_recommend(alg{i}, data, 'test_ratio', 0.2, 'times', 5);
        results{i} = outputs_t;
    end
    save(filename, 'results','-append');
end


%parpool('local', 4);
%addpath(genpath('~/code/recsys'))
%addpath(genpath('~/code/recsys2'))

%load ~/result/dmf_results(new_ndcg).mat
%paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
%clear('result');
%para = paras{2}; para = cell2struct(para(2:2:end), para(1:2:end),1);
%load ~/data/amazon/map/amazondata.mat
%item_feat = tfidf(item_feat);
%metric_fun = @(metric) metric.item_ndcg_score(1,end);
%alg = @(mat,varargin) dmf_aux_reg(mat, 'reg', true, 'K', 64, 'Y', item_feat, 'max_iter', 20, 'rho', para.rho, 'init', true, varargin{:});
%[outputs_reg{1:4}] = hyperp_search(@(varargin) rating_recommend(alg, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'beta_init', [0,1,10,50,100,500,1000]);
%save('~/data/feat_tuning.mat', 'outputs_reg');

