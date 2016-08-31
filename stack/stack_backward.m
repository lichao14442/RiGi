function stack = stack_backward(stack, ops, delta)
% backward of stackwork
% first used to CNN
% input:
% model,
% ops, option
% y: labels
% lichao, 20160717

%% params
layer_num = stack.layer_num;

%%  backprop deltas
for idx_layer = layer_num : -1 : 1

    if strcmp(stack.layers{idx_layer}.class, 'unit') % cost
        stack.layers{idx_layer} = unit_backward(stack.layers{idx_layer}, ops, delta);    
        
    elseif strcmp(stack.layers{idx_layer}.class, 'stack') % Affine
        stack.layers{idx_layer} = stack_backward(stack.layers{idx_layer}, ops, delta); 
    end
    
    delta = stack.layers{idx_layer}.delta;
    
end

stack.delta = stack.layers{1}.delta;
if isfield(stack.layers{layer_num},'costv')
    stack.costv = stack.layers{layer_num}.costv;
end
