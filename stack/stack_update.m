function stack = stack_update(stack, ops)
% used to update the network 
% first used in CNN
% input:
%       stackwork: stack
%       opts: option
% lichao, 20160717

%% params
layer_num = stack.layer_num;

%%
for idx_layer = 1 : layer_num
    if strcmp(stack.layers{idx_layer}.update, 'false') 
        continue;
    end
%         
    if strcmp(stack.layers{idx_layer}.class, 'unit') % 
        stack.layers{idx_layer} = unit_update(stack.layers{idx_layer}, ops);    
        
    elseif strcmp(stack.layers{idx_layer}.class, 'stack') %
        stack.layers{idx_layer} = stack_update(stack.layers{idx_layer}, ops);   
        
    end
end


end
