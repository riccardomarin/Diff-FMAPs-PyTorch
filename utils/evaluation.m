clear all;
model_name = 'log_model';
epoch = '4427';

load('log_model_4427_FAUST_noise_0.01.mat');

rng(3);
src = randperm(100);
tar = src(randperm(100));
assert(not(sum(src == tar)));

thr = [0:0.001:0.5];

match = zeros(100,1000);
match_opt = zeros(100,1000);

for i = 1:size(src,2)
    phiM = [ones(size(basis,2),1),squeeze(basis(src(i),:,:))];
    phiN = [ones(size(basis,2),1),squeeze(basis(tar(i),:,:))];
    [match(i,:), match_opt(i,:)] = our_match(phiM, phiN,[1:1000]);
    if(exist('desc'))
     descM = squeeze(desc(src(i),:,:));
     descN = squeeze(desc(tar(i),:,:));
     [match_desc(i,:)] = our_match_desc(phiM, phiN,descM, descN);
    end
end

mean_error = [];
for i = 1:100
    M.VERT = squeeze(vertices_clean(idx_src,:,:)); M.TRIV = double(faces); M.n = size(M.VERT,1); M.m = size(M.TRIV,1);
    N.VERT = squeeze(vertices_clean(idx_tar,:,:)); N.TRIV = double(faces); N.n = size(N.VERT,1); N.m = size(N.TRIV,1);

    dist_m = D;
    my_match = match_descour(i,:); match_n = match_desc10(i,:); match_3d = match_desc20(i,:); match_u = match_desc40(i,:);
    errors = compute_err(dist_m, [1:1000],[my_match', match_n',match_3d',match_u',[1:1000]']);
    errors = mean(errors,1);
    if i == 1
        %curves = compute_all_curves(errors,thr);
        mean_error(i,:,:) = errors;
    else
        %curves = curves + compute_all_curves(errors,thr);
        mean_error(i,:,:) = errors;
    end
end


save('match_descour','match_desc');
%%
clear all
load('match_desc10.mat')
match_desc10=match_desc;
load('match_desc20.mat');
match_desc20=match_desc;
load('match_desc40.mat');
match_desc40=match_desc;
load('match_descour.mat');
match_descour=match_desc;

load('FAUST_noise_0.01.mat');
load('N_calc.mat');

rng(3);
idx_src = randperm(100);
idx_tar = idx_src(randperm(100));


mean_error = [];
for i = 1:100
M.VERT = squeeze(vertices_clean(idx_src,:,:)); M.TRIV = double(faces); M.n = size(M.VERT,1); M.m = size(M.TRIV,1);
N.VERT = squeeze(vertices_clean(idx_tar,:,:)); N.TRIV = double(faces); N.n = size(N.VERT,1); N.m = size(N.TRIV,1);

dist_m = D;
my_match = match_descour(i,:); match_n = match_desc10(i,:); match_3d = match_desc20(i,:); match_u = match_desc40(i,:);
errors = compute_err(dist_m, [1:1000],[my_match', match_n',match_3d',match_u',[1:1000]']);
errors = mean(errors,1);
if i == 1
    %curves = compute_all_curves(errors,thr);
    mean_error(i,:,:) = errors;
else
    %curves = curves + compute_all_curves(errors,thr);
    mean_error(i,:,:) = errors;
end
end

mean(mean_error)

