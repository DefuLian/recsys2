function topkmat = topk_finder(train, U, V, topk, is_binary)
if is_binary
    U = compactbit(U > 0);
    V = compactbit(V > 0);
end
topkmat = topk_finder_(train, U, V, topk);
end
function topkmat = topk_finder_(train, U, V, topk)
    Et = train.';
    M = size(train, 1);
    step = 1000;
    num_step = floor((M + step-1)/step);
    topk_cell = cell(num_step, 1);
    for i=1:num_step
        start_u = (i-1)*step +1;
        end_u = min(i * step, M);
        subU = U(start_u:end_u, :); 
        subE = Et(:, start_u:end_u);
        subR_E = multiply(V, subU); 
        subR_E(subE ~= 0) = -inf;
        [~, Index] = maxk(subR_E, topk);
        [~, I, J] = find(Index);
        topk_cell{i} = [I + (i-1)*step, J, subR_E(sub2ind(size(subR_E), J, I))];
    end
    topkmat = cell2mat(topk_cell);
end
function out = multiply(a, b)
D = size(a,2) * 32;
if isa(a, 'uint32') && isa(b, 'uint32')
    %out = D-2*hamming_dist(a, b);
    out = 0.5 - hamming_dist(a, b)/D;
else
    %out = mult_real(a, b);
    out = mult_real(a, b)/(2*D);
end
end