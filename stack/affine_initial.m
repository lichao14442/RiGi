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
need_convert_dim = affine_model.need_convert_dim;
%% initial for each layer
linear_conf.indim = indim;
linear_conf.outdim = outdim;
linear_conf.name = [affine_model.name,'->linear'];
linear_conf.need_convert_dim = need_convert_dim;
linear_model = linear_set(linear_conf);
linear_model = linear_initial(linear_model);
%
bias_conf.indim = outdim;
bias_conf.name = [affine_model.name,'->bias'];
bias_model = bias_set(bias_conf);
bias_model = bias_initial(bias_model);
%
% set_stack
layers = {linear_model, bias_model};
layer_num = numel(layers);

%% (3) put into the struct
affine_model.layer_num = layer_num;
affine_model.layers = layers;
affine_model.type = 'affine';
affine_model.class = 'stack';
affine_model.update = 'true';
affine_model.dim = 1;

end
