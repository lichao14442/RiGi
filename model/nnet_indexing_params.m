function index_params = nnet_indexing_params(nnet)
% get the index of params
% lichao, 20160831

layer_num = nnet.layer_num;

%% 
% param_count = 1;
index_params = [];
for idx_layer = 1 : layer_num   %  for each layer
    if ~isfield(nnet.layers{idx_layer},'Params') ...
        && strcmp(nnet.layers{idx_layer}.class, 'unit')
        continue;
    end
    idx_layer_str = num2str(idx_layer);
    %
    if strcmp(nnet.layers{idx_layer}.class, 'unit') % cost
        idx_params_loc = {idx_layer_str};
    elseif strcmp(nnet.layers{idx_layer}.class, 'stack') % Affine
        idx_params_loc = [];
        idx_params_loc = stack_indexing_params(nnet.layers{idx_layer}, idx_params_loc, idx_layer_str);   
    end
    %
    index_params = [index_params idx_params_loc];
end