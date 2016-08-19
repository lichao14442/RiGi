function nnet = nnet_setup(nnet_conf)
% used to setup a nnet work
% first in CNN network
% 20160717 lichao
%
 

%
%% input layer
assert (strcmp(nnet_conf{1}.type, 'input'), 'the first layer type must be input ');
inmaps_num = 1;
inputlayer = nnet_conf{1};
inputlayer.inmaps_num = inmaps_num;
inputlayer.outmaps_num = inmaps_num;
inputlayer.outmap_size = inputlayer.inmap_size;
inputlayer.outdim = inputlayer.indim;



nnet.layers{1} = inputlayer;
nnet.struct = 'input';
% batch_size = inputlayer.batch_size;
%% for hide layers
layer_num = numel(nnet_conf);
for idx_layer = 2 : layer_num   %  layer
%     if strcmp(nnet_conf{idx_layer}.type, 'conv2d') || strcmp(nnet_conf{idx_layer}.type, 'pool2d') ...
%             || strcmp(nnet_conf{idx_layer}.type, 'conv2dpack') || strcmp(nnet_conf{idx_layer}.type, 'pool2dpack')%% 2D keep
%         inmaps_num = nnet.layers{idx_layer-1}.outmaps_num;
%         inmap_size = nnet.layers{idx_layer-1}.outmap_size;
%     elseif isfield(nnet_conf{idx_layer}, 'need_convert_dim') && strcmp(nnet_conf{idx_layer}.need_convert_dim, 'true')  % 2D -> 1D
%         indim = prod(nnet.layers{idx_layer-1}.outmap_size) *  nnet.layers{idx_layer-1}.outmaps_num;
%     else        % 1D keep
%         indim = nnet.layers{idx_layer-1}.outdim;  
%     end
    
    switch nnet_conf{idx_layer}.type
        case 'conv2d'
            conv2d_conf = nnet_conf{idx_layer}; 
            conv2d_conf.inmaps_num = nnet.layers{idx_layer-1}.outmaps_num;
            conv2d_conf.inmap_size = nnet.layers{idx_layer-1}.outmap_size;
            %
            conv2d_model = convolution2d_set(conv2d_conf);
            conv2d_model = convolution2d_initial(conv2d_model);
            nnet.layers{idx_layer} = conv2d_model;
            nnet.struct = [nnet.struct, ' ', conv2d_model.name];

        case  'pool2d' 
            pool2d_conf = nnet_conf{idx_layer};
            pool2d_conf.inmaps_num = nnet.layers{idx_layer-1}.outmaps_num;
            pool2d_conf.inmap_size = nnet.layers{idx_layer-1}.outmap_size;
            %
            pool2d_model = pooling2d_set(pool2d_conf);
            pool2d_model = pooling2d_initial(pool2d_model);
            nnet.layers{idx_layer} = pool2d_model;
            nnet.struct = [nnet.struct, ' ', pool2d_model.name];

            
       case 'conv2dpack'
            conv2dpack_conf = nnet_conf{idx_layer}; 
            conv2dpack_conf.inmaps_num = nnet.layers{idx_layer-1}.outmaps_num;
            conv2dpack_conf.inmap_size = nnet.layers{idx_layer-1}.outmap_size;
            %
            conv2dpack_model = conv2dPackage_set(conv2dpack_conf);
            conv2dpack_model = conv2dPackage_initial(conv2dpack_model);
            nnet.layers{idx_layer} = conv2dpack_model;
            nnet.struct = [nnet.struct, ' ', conv2dpack_model.name];
            
       case 'pool2dpack'
            pool2dpack_conf = nnet_conf{idx_layer}; 
            pool2dpack_conf.inmaps_num = nnet.layers{idx_layer-1}.outmaps_num;
            pool2dpack_conf.inmap_size = nnet.layers{idx_layer-1}.outmap_size;
            %
            pool2dpack_model = pool2dPackage_set(pool2dpack_conf);
            pool2dpack_model = pool2dPackage_initial(pool2dpack_model);
            nnet.layers{idx_layer} = pool2dpack_model;
            nnet.struct = [nnet.struct, ' ', pool2dpack_model.name];
            
        case 'full'
            full_conf = nnet_conf{idx_layer};
            full_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            full_model = fullLinear_set(full_conf);
            full_model = fullLinear_initial(full_model);
            nnet.layers{idx_layer} = full_model;
            nnet.struct = [nnet.struct, ' ', full_model.name];
            
        case  'nonlinear'
            nonlinear_conf = nnet_conf{idx_layer};
            nonlinear_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            nonlinear_model = nonlinear_set(nonlinear_conf);
            nonlinear_model = nonlinear_initial(nonlinear_model);
            nnet.layers{idx_layer} = nonlinear_model;
            nnet.struct = [nnet.struct, ' ', nonlinear_model.name];
            
        case  'batchnorm'
            batchnorm_conf = nnet_conf{idx_layer};
            batchnorm_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            batchnorm_model = batchnorm_set(batchnorm_conf);
            batchnorm_model = batchnorm_initial(batchnorm_model);
            nnet.layers{idx_layer} = batchnorm_model;
            nnet.struct = [nnet.struct, ' ', batchnorm_model.name];  
            
        case 'ce-cost'
            ce_cost_conf = nnet_conf{idx_layer};
            ce_cost_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            ce_cost_model = ce_cost_set(ce_cost_conf);
            ce_cost_model = ce_cost_initial(ce_cost_model);
            nnet.layers{idx_layer} = ce_cost_model;
            nnet.struct = [nnet.struct, ' ', ce_cost_model.name];
            
        case 'mse-cost'
            mse_cost_conf = nnet_conf{idx_layer};
            mse_cost_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            mse_cost_model = mse_cost_set(mse_cost_conf);
            mse_cost_model = mse_cost_initial(mse_cost_model);
            nnet.layers{idx_layer} = mse_cost_model;
            nnet.struct = [nnet.struct, ' ', mse_cost_model.name];
            
        case  'affine'
            affine_conf = nnet_conf{idx_layer};
            affine_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            affine_model = affine_set(affine_conf);
            affine_model = affine_initial(affine_model);
            nnet.layers{idx_layer} = affine_model;
            nnet.struct = [nnet.struct, ' ', affine_model.name];
            
        case  'fullconnect'
            full_conf = nnet_conf{idx_layer};
            full_conf.indim = nnet.layers{idx_layer-1}.outdim;
            %
            full_model = fullconnect_set(full_conf);
            full_model = fullconnect_initial(full_model);
            nnet.layers{idx_layer} = full_model;
            nnet.struct = [nnet.struct, ' ', full_model.name];
        otherwise
            error('the type of layer is UNDIFEND');
    end
    
end
nnet.layer_num = layer_num;
end
