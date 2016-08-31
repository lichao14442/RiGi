function index_params = stack_indexing_params(stack, index_params, idx_upper_str)
% get the index of params of stack layer 
% index_params: '1->2->1'
% lichao , 20160831

layer_num = stack.layer_num;

%% forward 
original_str = idx_upper_str;
for idx_layer = 1 : layer_num   %  for each layer in stack
    if ~isfield(stack.layers{idx_layer},'Params') ...
        && strcmp(stack.layers{idx_layer}.class, 'unit')
        continue;
    end
    idx_layer_str = [original_str, '->',num2str(idx_layer)];
    if strcmp(stack.layers{idx_layer}.class, 'unit') % cost
        idx_params_loc = {idx_layer_str};    
        
    elseif strcmp(stack.layers{idx_layer}.class, 'stack') % Affine
        idx_params_loc = [];
        idx_params_loc = stack_indexing_params(stack.layers{idx_layer}, idx_params_loc, idx_layer_str);   
    end
    
    index_params = [index_params idx_params_loc];
end

end
