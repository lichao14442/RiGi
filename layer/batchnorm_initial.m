function batchnorm_model = batchnorm_initial(batchnorm_model)
% used to initialize the fulllinear model
% params [in]
%    struct('type', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160725

%% used params to initialize
indim = batchnorm_model.indim;
axis_to_norm = batchnorm_model.axis_to_norm;
inmap_size = batchnorm_model.inmap_size;
inmaps_num = batchnorm_model.inmaps_num;

%%
if axis_to_norm == 0
    param_dim = indim;
    go_ops_order = [];
    back_ops_order = [];
else
    maps3d_size = [inmap_size(1),inmap_size(2),inmaps_num];
    param_dim = maps3d_size(axis_to_norm);
%     batch_size_new = prod(allmap_size)/param_dim;
    % go 
    go_ops_order = [ 1 2 3 4];
    go_ops_order(axis_to_norm) = [];
    go_ops_order = [axis_to_norm go_ops_order];
    % back
    back_ops_order = [ 2 3 4 0]; % 0 if dummpy
    idx = 4;
    while idx > axis_to_norm
        back_ops_order(idx) = back_ops_order(idx-1);
        idx = idx - 1;
    end
    back_ops_order(axis_to_norm) = 1;
    %
%     orderedmap_size = allmap_size(go_ops_order);
end

gamma = (rand(param_dim, 1) - 0.5) * 2 * sqrt(6 / (param_dim + 1));
beta =  zeros(param_dim, 1);
dbeta = zeros(param_dim, 1);
dgamma = zeros(param_dim, 1);
running_mean = zeros(param_dim, 1);
running_var = zeros(param_dim, 1);
eval_mean = zeros(param_dim, 1);
eval_var = zeros(param_dim, 1);
running_samples = 0;
running_iters = 0;

%% (3) put into the struct

batchnorm_model.outdim = indim;
batchnorm_model.outmap_size = inmap_size;
batchnorm_model.outmaps_num = inmaps_num;
batchnorm_model.param_dim = param_dim;
% batchnorm_model.batch_size_new = batch_size_new;
batchnorm_model.running_mean = running_mean;
batchnorm_model.running_var = running_var;
batchnorm_model.running_samples = running_samples;
batchnorm_model.running_iters = running_iters;
batchnorm_model.eval_mean = eval_mean;
batchnorm_model.eval_var = eval_var;
%
batchnorm_model.Params = {gamma, beta};
batchnorm_model.dParams = {dgamma, dbeta};
%
batchnorm_model.go_ops_order = go_ops_order;
batchnorm_model.back_ops_order = back_ops_order;
%
batchnorm_model.type = 'batchnorm';
batchnorm_model.class = 'unit';
batchnorm_model.update = 'true';
batchnorm_model.is_cost = 'false';
end
