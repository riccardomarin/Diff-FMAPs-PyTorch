function [err] = compute_err(dist_m,gt_match, matches)
    for i = 1:size(matches,2)
        for j = 1:size(matches,1)
            err(j,i) = dist_m(gt_match(j),matches(j,i));
        end
    end