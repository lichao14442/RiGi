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
else
    param_dim = inmaps_num;
end

gamma = single(rand(param_dim, 1) - 0.5) * 2 * sqrt(6 / (param_dim + 1));
beta =  single(zeros(param_dim, 1));
dbeta = single(zeros(param_dim, 1));
dgamma = single(zeros(param_dim, 1));
running_mean = single(zeros(param_dim, 1));
running_var = single(zeros(param_dim, 1));
eval_mean = single(zeros(param_dim, 1));
eval_var = single(zeros(param_dim, 1));
running_samples = 0;
running_iters = 0;
%% (3) put into the struct
if ~isfield(batchnorm_model, 'need_convert_dim')
    batchnorm_model.need_convert_dim = 'false';
end
%
if ~isfield(batchnorm_model, 'dropoutFraction')
    batchnorm_model.dropoutFraction = 0;
end

batchnorm_model.outdim = indim;
batchnorm_model.outmap_size = inmap_size;
batchnorm_model.outmaps_num = inmaps_num;
batchnorm_model.param_dim = param_dim;
batchnorm_model.gamma = gamma;
batchnorm_model.beta = beta;
batchnorm_model.dbeta = dbeta;
batchnorm_model.dgamma = dgamma;
batchnorm_model.running_mean = running_mean;
batchnorm_model.running_var = running_var;
batchnorm_model.running_samples = running_samples;
batchnorm_model.running_iters = running_iters;
batchnorm_model.eval_mean = eval_mean;
batchnorm_model.eval_var = eval_var;
%
batchnorm_model.type = 'batchnorm';
batchnorm_model.class = 'unit';
batchnorm_model.update = 'true';
batchnorm_model.dim = 1;
end
