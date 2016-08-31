function nnet = nnet_backward(nnet, ops, y)
% backward of nnetwork
% first used to CNN
% input:
% model,
% ops, option
% y: labels
% lichao, 20160717

%% params
layer_num = nnet.layer_num;
delta = y;
%%  backprop deltas
for idx_layer = layer_num : -1 : 1

    if strcmp(nnet.layers{idx_layer}.class, 'unit') % cost
        nnet.layers{idx_layer} = unit_backward(nnet.layers{idx_layer}, ops, delta);    
        
    elseif strcmp(nnet.layers{idx_layer}.class, 'stack') % Affine
        nnet.layers{idx_layer} = stack_backward(nnet.layers{idx_layer}, ops, delta); 
    end
    %
     delta = nnet.layers{idx_layer}.delta;
%     if strcmp(nnet.layers{idx_layer}.need_convert_dim, 'true')  % reshape
%         map_size  = nnet.layers{idx_layer-1}.outmap_size;
%         maps_num = nnet.layers{idx_layer-1}.outmaps_num;
%         delta = convert1d_to_2d(nnet.layers{idx_layer}.delta, map_size, maps_num);
%     else
%         delta = nnet.layers{idx_layer}.delta;
%     end
end

nnet.delta = nnet.layers{1}.delta;
nnet.costv = nnet.layers{layer_num}.costv;
nnet.err = nnet.layers{layer_num}.delta;

end
