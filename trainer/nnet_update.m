function nnet = nnet_update(nnet, ops)
% used to update the network 
% first used in CNN
% input:
%       nnetwork: nnet
%       opts: option
% lichao, 20160717

%% params
layer_num = nnet.layer_num;

%%
for idx_layer = 2 : layer_num
    if strcmp(nnet.layers{idx_layer}.update, 'false') 
        continue;
    end
%         
    if strcmp(nnet.layers{idx_layer}.class, 'unit') % 
        nnet.layers{idx_layer} = unit_update(nnet.layers{idx_layer}, ops);    
        
    elseif strcmp(nnet.layers{idx_layer}.class, 'stack') %
        nnet.layers{idx_layer} = stack_update(nnet.layers{idx_layer}, ops);   
        
    end
end


end
