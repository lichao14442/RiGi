function affine_model = affine_initial(affine_model)
% used to initialize the affine model
% params [in]
%    struct('name', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160718

%% used params to initialize
indim = affine_model.indim;
outdim = affine_model.outdim;
% need_convert_dim = affine_model.need_convert_dim;
batch_normalized = affine_model.batch_normalized;
%% initial for each layer
linear_conf.indim = indim;
linear_conf.outdim = outdim;
linear_conf.name = [affine_model.name,'->linear'];
% linear_conf.need_convert_dim = need_convert_dim;
linear_model = linear_set(linear_conf);
linear_model = linear_initial(linear_model);
%
if strcmp(batch_normalized, 'true')
    batchnorm_conf.indim = outdim;
    batchnorm_conf.name = [affine_model.name,'->batchnorm'];
    bias_model = batchnorm_set(batchnorm_conf);
    bias_model = batchnorm_initial(bias_model);
else
    bias_conf.indim = outdim;
    bias_conf.name = [affine_model.name,'->bias'];
    bias_model = bias_set(bias_conf);
    bias_model = bias_initial(bias_model);
end
%
% set_stack
layers = {linear_model, bias_model};

%% (3) put into the struct
affine_model = stack_build(layers, affine_model);
affine_model.type = 'affine';

end
