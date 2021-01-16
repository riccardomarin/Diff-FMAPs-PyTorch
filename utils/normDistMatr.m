function dist_m = normDistMatr(N)
    dist_m = calc_dist_matrix(N,[1:N.n]);
    diam = max(max(dist_m));
    dist_m = dist_m./diam;