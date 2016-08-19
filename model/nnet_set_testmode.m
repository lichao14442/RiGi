function nnet = nnet_set_testmode(nnet)
% for batchnormlization layer
% used first in CNN
% lichao , 20160726

layer_num = nnet.layer_num;

%% set testmode
for idx_layer = 2 : layer_num   %  for each layer
    if strcmp(nnet.layers{idx_layer}.class, 'unit') &&  strcmp(nnet.layers{idx_layer}.type, 'batchnorm') % cost
        nnet.layers{idx_layer}.test_mode = 'true';    
        
    elseif strcmp(nnet.layers{idx_layer}.class, 'stack') % Affine
        nnet.layers{idx_layer} = nnet_set_testmode(nnet.layers{idx_layer});   
     
    end
end

end
