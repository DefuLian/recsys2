function [eval_summary, eval_detail, elapsed] = recommender(rec_hash, rec_real, data, varargin)
[eval_summary, eval_detail, elapsed] = heldout_rec(rec_hash, rec_real, data, varargin{:});
end

function [eval_summary, eval_detail, elapsed] = heldout_rec(rec_hash, rec_real, mat, varargin)
[test_ratio, split_mode, times, seed,topk, cutoff, isbinary] = process_options(varargin, ...
    'test_ratio', 0.2, 'split_mode', 'un', 'times', 1, 'seed', 1, 'topk', 500, ...
    'cutoff',200, 'isbinary', true);

    elapsed = zeros(times,2);

    rng(seed);
    eval_detail = struct();
    metric_times = cell(times,1);
    tests = cell(times,1);
    for t=1:times
        [~, tests{t}] = split_matrix(mat, split_mode, 1-test_ratio);
    end

    for t=1:times
        test = tests{t}; train = sparse(mat - test);
        fprintf('%d,%d\n',full(sum(train(:))),full(sum(test(:))));
        tic; [B, D] = rec_hash(train);
        %[P, Q] = rec_real(train, 'P0',B, 'Q0',D, 'usage', 'concate'); t1 = toc;
        [P, Q] = rec_real(train); t1 = toc;
        tic;  metric_times{t} = compute_score(); t2 = toc;
        elapsed(t, :) = [t1,t2];
    end

    for t=1:times
        metric_time = metric_times{t};
        fns = fieldnames(metric_time);
        for f=1:length(fns)
            fieldname = fns{f};
            if isfield(eval_detail, fieldname)
                %evalout.(fieldname) = evalout.(fieldname) + [metric_time.(fieldname);(metric_time.(fieldname)).^2];
                eval_detail.(fieldname) = [eval_detail.(fieldname); metric_time.(fieldname)];
            else
                %evalout.(fieldname) = [metric_time.(fieldname);(metric_time.(fieldname)).^2];
                eval_detail.(fieldname) = metric_time.(fieldname);
            end
        end
    end
    fns = fieldnames(eval_detail);
    for f=1:length(fns)
        fieldname = fns{f};
        field = eval_detail.(fieldname);
        %field_mean = field(1,:) / times;
        %field_std = sqrt(field(2,:)./times - field_mean .* field_mean);
        eval_summary.(fieldname) = [mean(field,1); std(field,0,1)];
    end
    elapsed = mean(elapsed, 1);

function eval = compute_score()
    test(train~=0) = 0;
    cand_count = full(size(train,2) - sum(train~=0,2)); 
    tuples = topk_finder(train, B, D, topk, isbinary); %return topk items for evalaution of next stage%
    [mat_rank, user_count, ~] = predict_tuple(tuples, test, P, Q, cutoff);
    if ~isexplit(test)
        eval = compute_item_metric(mat_rank, user_count, cand_count, topk, cutoff);
    else
        eval = compute_rating_metric(test, mat_rank, cand_count, topk, cutoff);
    end
end
end

