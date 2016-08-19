function stack = stack_forward(stack, x)
% forward of stack 
% used first in CNN
% lichao , 20160718

layer_num = stack.layer_num;

%% forward 
input = x;
for idx_layer = 1 : layer_num   %  for each layer in stack

    if strcmp(stack.layers{idx_layer}.class, 'unit') % cost
        stack.layers{idx_layer} = unit_forward(stack.layers{idx_layer}, input);    
        
    elseif strcmp(stack.layers{idx_layer}.class, 'stack') % Affine
        stack.layers{idx_layer} = stack_forward(stack.layers{idx_layer}, input);
    end
    
    input = stack.layers{idx_layer}.h;
    
end
stack.x = x;
stack.h = stack.layers{layer_num}.h;


end
