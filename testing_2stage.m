parpool('local', 3);
addpath(genpath('~/code/recsys/'))
addpath(genpath('~/code/recsys2/'))
dir = '~/data';
datasets = {'yelpdata', 'amazondata', 'ml10Mdata', 'netflixdata'};
load('~/result/dmf_results(new_ndcg).mat')
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
clear('result');
file_name = sprintf('%s/two_stage_results.mat',dir);
if exist(file_name, 'file')
    load(file_name);
end
if ~exist('result', 'var')
    result = cell(length(datasets),5);
end

for i=1:length(datasets)
    if ~isempty(result{i,1})
        continue
    end
    dataset = datasets{i};
    para = paras{i}; para = cell2struct(para(2:2:end), para(1:2:end),1);
    fprintf('%s\n', dataset)
    load(sprintf('%s/%s.mat', dir , dataset))
    if ~exist('data','var')
        data = Traindata + Testdata;
    end
    rec_hash = @(mat) dmf(mat, 'K', 64, 'max_iter',20, 'rho',para.rho, 'alpha', para.alpha, 'beta',para.beta);
    rec_real = @(mat, varargin) gmf(mat, 'K', 64, 'max_iter',20, 'rho',para.rho, 'eta', 0.01, varargin{:});
    topks = [500,1000,1500,2000,2500,3000];
    for tk = 1:length(topks)
        [output{1:3}] = recommender(rec_hash, rec_real, data, 'test_ratio', 0.2, 'topk', topks(tk), 'cutoff', 200);
        result{i,tk} = output;
    end
    if isexplit(data)
        [output{1:3}] = rating_recommend(rec_real, data, 'test_ratio', 0.2, 'topk', 200);
        result{i,tk+1} = output;
    else
        [output{1:3}] = item_recommend(rec_real, data, 'test_ratio', 0.2, 'topk', 200);
        result{i,tk+1} = output;
    end
    save(file_name, 'result');
    clear('data')
end
