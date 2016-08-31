function [Params, dParams] = stack_extract_params(stack, index_params)
% according the index_params to extract the params in the index
% index_params: '1->2->1'
% lichao, 20160831



layer = stack;
idx = regexp(index_params,'->','split');

deep_params = length(idx);

for i = 1: deep_params
    layer = layer.layers{str2double(idx{i})};
end

Params = layer.Params;
dParams = layer.dParams;
end
