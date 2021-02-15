clear all;
addpath(genpath('utils'));
addpath(genpath('evaluation'));


% Load Basis and descriptors
%load(['log_model_4427_FAUST_noise_0.01.mat']);
load(['out_FAUST_noise_0.01.mat']);

% Set the evaluation couples
rng(3);
src = randperm(100);
tar = src(randperm(100));
assert(not(sum(src == tar)));

% Computing the matching for each couple
match = zeros(100,1000);
match_opt = zeros(100,1000);

for i = 1:size(src,2)
    phiM = [squeeze(basis(src(i),:,:))];
    phiN = [squeeze(basis(tar(i),:,:))];
    [match(i,:), match_opt(i,:)] = our_match(phiM, phiN,[1:1000]);
    if(exist('desc'))
     descM = squeeze(desc(src(i),:,:));
     descN = squeeze(desc(tar(i),:,:));
     [match_desc(i,:)] = our_match_desc(phiM, phiN,descM, descN);
    else
        match_desc(i,:) = match_opt(i,:);
    end
end


%% Evaluation

mean_error = [];

% We load the geodesic distance matrix
load('./utils/N_out.mat');
thr = [0:0.0001:0.5];

for i = 1:100
    idx_src = src(i); idx_tar = tar(i);
    
    M.VERT = squeeze(vertices_clean(src,:,:)); M.TRIV = double(faces); M.n = size(M.VERT,1); M.m = size(M.TRIV,1);
    N.VERT = squeeze(vertices_clean(tar,:,:)); N.TRIV = double(faces); N.n = size(N.VERT,1); N.m = size(N.TRIV,1);

    dist_m = D;
    match_d = match_desc(i,:); match_o = match_opt(i,:);
    errors = compute_err(dist_m, [1:1000],[match_d', match_o']);
    if i == 1
        curves = compute_all_curves(errors,thr);
        mean_error = errors;
    else
        curves = curves + compute_all_curves(errors,thr);
        mean_error = mean_error + errors;
    end
end

mean_curves = curves/size(src,2);
mean_error = mean(mean_error/size(src,2));

plot(thr,mean_curves,'LineWidth',2);
legend(['Our:', num2str(mean_error(1),2)], ['Our+Opt:', num2str(mean_error(2),2)]);
set(gcf,'color','w');

