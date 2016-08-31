function nnet = nnet_forward(nnet, x)
% forward of network
% used first in CNN
% lichao , 20160717

layer_num = nnet.layer_num;

%% forward 
input = x;
for idx_layer = 1 : layer_num   %  for each layer
%     if strcmp(nnet.layers{idx_layer}.need_convert_dim, 'true') 
%         input = convert2d_to_1d(nnet.layers{idx_layer-1}.h);
%     else
%         input = nnet.layers{idx_layer-1}.h;
%     end
    
    %
    if strcmp(nnet.layers{idx_layer}.class, 'unit') % cost
        nnet.layers{idx_layer} = unit_forward(nnet.layers{idx_layer}, input);    
        
    elseif strcmp(nnet.layers{idx_layer}.class, 'stack') % Affine
        nnet.layers{idx_layer} = stack_forward(nnet.layers{idx_layer}, input);   
     
    end
    input = nnet.layers{idx_layer}.h;
end
nnet.x = x;
nnet.h = nnet.layers{layer_num}.h;


end
