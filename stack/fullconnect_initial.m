function full_model = fullconnect_initial(full_model)
% used to initialize the affine model
% params [in]
%    struct('name', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160718

%% used params to initialize
indim = full_model.indim;
outdim = full_model.outdim;
nonlinearity = full_model.nonlinearity;
need_convert_dim = full_model.need_convert_dim;
batch_normalized = full_model.batch_normalized;

%% initial for each layer
affine_conf.indim = indim;
affine_conf.outdim = outdim;
affine_conf.batch_normalized = batch_normalized;
affine_conf.need_convert_dim = need_convert_dim;
affine_conf.name = [full_model.name,'->affine'];
affine_model = affine_set(affine_conf);
affine_model = affine_initial(affine_model);
%
nonlinear_conf.indim = outdim;
nonlinear_conf.nonlinearity = nonlinearity;
nonlinear_conf.name = [full_model.name,'->nonlinear'];
nonlinear_model = nonlinear_set(nonlinear_conf);
nonlinear_model = nonlinear_initial(nonlinear_model);
%
% set_stack
layers = {affine_model, nonlinear_model};
layer_num = numel(layers);

%% (3) put into the struct
full_model.layer_num = layer_num;
full_model.layers = layers;
full_model.type = 'fullcontection';
full_model.class = 'stack';
full_model.update = 'true';
affine_model.dim = 1;
end
