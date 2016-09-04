function conv2dpack_model = conv2dPackage_initial(conv2dpack_model)
% used to initialize the conv2d model
% params [in]
%    struct('name', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160728

%% used params to initialize
inmaps_num = conv2dpack_model.inmaps_num;
outmaps_num = conv2dpack_model.outmaps_num;
inmap_size = conv2dpack_model.inmap_size;
kernelsize = conv2dpack_model.kernelsize;
is_same_size = conv2dpack_model.is_same_size;
nonlinearity = conv2dpack_model.nonlinearity;
batch_normalized = conv2dpack_model.batch_normalized;
order = conv2dpack_model.order;

%% initial for each layer
%(1) conv2d
conv2d_conf.inmaps_num = inmaps_num;
conv2d_conf.inmap_size = inmap_size;
conv2d_conf.is_same_size = is_same_size;
conv2d_conf.kernelsize = kernelsize;
conv2d_conf.outmaps_num = outmaps_num;
conv2d_conf.order = order;
conv2d_conf.name = [conv2dpack_model.name,'->conv2d'];

conv2d_model = convolution2d_set(conv2d_conf);
conv2d_model = convolution2d_initial(conv2d_model);
%
% (2) batchnorm || bias
outmap_size = conv2d_model.outmap_size;
outdim = outmaps_num * prod(outmap_size);
biaslayer_conf.indim = outdim;
biaslayer_conf.inmaps_num = outmaps_num;
biaslayer_conf.inmap_size = outmap_size;
biaslayer_conf.axis_to_norm = strfind(order,'c');
if strcmp(batch_normalized,'true')
    biaslayer_conf.name = [conv2dpack_model.name,'->batchnorm'];
    biaslayer_model = batchnorm_set(biaslayer_conf);
    biaslayer_model = batchnorm_initial(biaslayer_model);
else
    biaslayer_conf.name = [conv2dpack_model.name,'->bias'];
    biaslayer_model = bias_set(biaslayer_conf);
    biaslayer_model = bias_initial(biaslayer_model);
end

% (3) unlineaer
nonlinear_conf.indim = outdim;
nonlinear_conf.nonlinearity = nonlinearity;
nonlinear_conf.inmaps_num = outmaps_num;
nonlinear_conf.inmap_size = outmap_size;
nonlinear_conf.name = [conv2dpack_model.name,'->nonlinear'];

nonlinear_model = nonlinear_set(nonlinear_conf);
nonlinear_model = nonlinear_initial(nonlinear_model);
%
layers = {conv2d_model, biaslayer_model, nonlinear_model};

%% (3) put into the struct
conv2dpack_model = stack_build(layers, conv2dpack_model);
conv2dpack_model.type = 'conv2dpack';

end
