function R = scale_matrix(R, s)
maxS = max(max(R));
minS = min(R(R~=0));
[I, J, V] = find(R);
if maxS ~= minS
    VV = (V-minS)/(maxS-minS);
    VV = 2 * s * VV - s + 1e-10;
else
    VV = V .* s ./ maxS;
end
R = sparse(I, J, VV, size(R,1), size(R,2));
end