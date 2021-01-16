function c = compute_all_curves(err,thr)
    for i=1:size(err,2)
        c(i,:) = calc_err_curve(err(:,i),thr);
    end