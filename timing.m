%tic;a = topk_finder(train, test, B, D, 500, true);toc;
[output{1:3}] = recommender(rec_hash, rec_real, data, 'test_ratio', 0.2, 'topk', 1000, 'cutoff', 200);