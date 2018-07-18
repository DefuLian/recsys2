parpool('local', 3);
addpath(genpath('~/code/recsys/'))
addpath(genpath('~/code/recsys2/'))
dir = '~/data';

load('~/result/dmf_results(new_ndcg).mat')
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
para = paras{2}; para = cell2struct(para(2:2:end), para(1:2:end),1);
clear('result');

ll = 6;
file_name = sprintf('%s/two_stage_item_size_results.mat',dir);
if exist(file_name, 'file')
    load(file_name);
end
if ~exist('result', 'var')
    result = cell(ll, 3);
end

data = readContent(sprintf('%s/amazon/ratings_mapped.csv', dir),'sep',',');
data2 = data;

rec_hash = @(mat) dmf(mat, 'K', 64, 'max_iter',20, 'rho', para.rho, 'alpha', para.alpha, 'beta',para.beta);
rec_real = @(mat, varargin) gmf(mat, 'K', 64, 'max_iter',20, 'rho', para.rho, 'eta', 0.01, varargin{:});
sz = 2:2:20;
for s = 1:ll
    %if ~isempty(result{s, 1})
    %    continue
    %end
    fprintf('%d\n',sz(s));
    data = trim_data(data2, sz(s));
    result{s, 3} = size(data,2);
    [output{1:3}] = recommender(rec_hash, rec_real, data, 'test_ratio', 0.2, 'topk', 1000, 'cutoff', 200);
    result{s, 1} = output;
    if isexplit(data)
        [output{1:3}] = rating_recommend(rec_real, data, 'test_ratio', 0.2, 'topk', 200);
        result{s,2} = output;
    else
        [output{1:3}] = item_recommend(rec_real, data, 'test_ratio', 0.2, 'topk', 200);
        result{s,2} = output;
    end
    save(file_name, 'result');
end


