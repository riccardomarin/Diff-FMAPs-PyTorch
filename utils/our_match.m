function [match, match_opt] = our_match(phiM,phiN, gt_match,gt_matchM)
    match = knnsearch(phiN,phiM);
    if exist('gt_matchM')
        C = phiM(gt_matchM,:)\phiN(gt_match,:);
    else
        C = phiM\phiN(gt_match,:);
    end
    match_opt = knnsearch(phiN,phiM*C);

