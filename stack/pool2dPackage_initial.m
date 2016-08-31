function pool2dpack_model = pool2dPackage_initial(pool2dpack_model)
% used to initialize the pool2d model
% params [in]
%    struct('name', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160728

%% used params to initialize
inmaps_num = pool2dpack_model.inmaps_num;
outmaps_num = pool2dpack_model.outmaps_num;
inmap_size = pool2dpack_model.inmap_size;
scale = pool2dpack_model.scale;
batch_normalized = pool2dpack_model.batch_normalized;
order = pool2dpack_model.order;

%% initial for each layer
%(1) pool2d
pool2d_conf.inmaps_num = inmaps_num;
pool2d_conf.inmap_size = inmap_size;
pool2d_conf.scale = scale;
pool2d_conf.order = order;
pool2d_conf.name = [pool2dpack_model.name,'->pool2d'];

pool2d_model = pooling2d_set(pool2d_conf);
pool2d_model = pooling2d_initial(pool2d_model);
outmap_size = pool2d_model.outmap_size;
%
% (2) batchnorm
outdim = prod(outmap_size)*outmaps_num;
if strcmp(batch_normalized,'true')
    batchnorm_conf.indim = outdim;
    batchnorm_conf.name = [pool2dpack_model.name,'->batchnorm'];
    %
    batchnorm_model = batchnorm_set(batchnorm_conf);
    batchnorm_model = batchnorm_initial(batchnorm_model); 
    layers = {pool2d_model, batchnorm_model};
else
    layers = {pool2d_model};
end

% (last) set_stack
layer_num = numel(layers);

%% (3) put into the struct
pool2dpack_model.outdim = outdim;
pool2dpack_model.outmaps_num = outmaps_num;
pool2dpack_model.outmap_size = outmap_size;
pool2dpack_model.layer_num = layer_num;
pool2dpack_model.layers = layers;
pool2dpack_model.type = 'pool2dpack';
pool2dpack_model.class = 'stack';
pool2dpack_model.update = 'true';
pool2dpack_model.is_cost = 'false'; 

end
