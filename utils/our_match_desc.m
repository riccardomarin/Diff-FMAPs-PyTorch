function [match] = our_match_desc(phiM,phiN, descM, descN)
    F = pinv(phiM)*descM; %T
    G = pinv(phiN)*descN; %C
    C = (F'\G')'; %T\C
    
    match = knnsearch(phiN*C, phiM);

    
    %     match = knnsearch(phiN*C', phiM); 0,54
    %     match = knnsearch(phiN, phiM*C) 0,6
    %     match = knnsearch(phiN, phiM*C'); 0,5
    %     match = knnsearch(phiN*C, phiM); worst