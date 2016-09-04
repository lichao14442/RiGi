function stack = stack_build(layers_cell,stack)
% put multi layers into a stack struct
% lichao, 20160901


%TODO: check legal, output size of parent is equal the input size of child
%layer
if nargin < 2
    stack = [];
end

stack.layers = layers_cell;
stack.layer_num = length(stack.layers);
stack.type = [];
for i = 1: stack.layer_num
    stack.type = [stack.type, ' ',stack.layers{i}.type];
end
stack.type(1) = [];
stack.type = ['< ',stack.type,' >'];
stack.class = 'stack';

stack.is_cost = stack.layers{end}.is_cost;
stack.outdim = stack.layers{end}.outdim;
%
if isfield(stack.layers{end},'outmap_size')
    stack.outmap_size = stack.layers{end}.outmap_size;
else
    stack.outmap_size = [1 1];
end
%
if isfield(stack.layers{end},'outmaps_num')
    stack.outmaps_num = stack.layers{end}.outmaps_num;
else
    stack.outmaps_num = stack.outdim/prod(stack.outmap_size);
end
%


stack.update = 'false';
for i = 1: stack.layer_num
    if strcmp(stack.layers{i}.update,'true') 
        stack.update = 'true';
        break;
    end
end
