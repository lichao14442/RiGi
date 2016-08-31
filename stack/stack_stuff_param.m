function stack = stack_stuff_param(stack, index_params,  param, count_param)
% according the index_params and count_param to stuff the params
% index_params: '1->2->1'
% count_param: 1  | 2 | 3 ...
% lichao, 20160831

if nargin < 4
    count_param = 1;
end

%%
idx = regexp(index_params,'->','split');
deep_params = length(idx);
idx = str2double(idx);
switch (deep_params)
    case 1
        stack.layers{idx(1)}.Params{count_param} = param;
    case 2
        stack.layers{idx(1)}.layers{idx(2)}.Params{count_param} ...
            = param;
    case 3
        stack.layers{idx(1)}.layers{idx(2)}.layers{idx(3)}.Params{count_param} ...
            = param;
    case 4
        stack.layers{idx(1)}.layers{idx(2)}.layers{idx(3)}.layers{idx(4)}.Params{count_param} ...
            = param;
    case 5
        stack.layers{idx(1)}.layers{idx(2)}.layers{idx(3)}.layers{idx(4)}.layers{idx(5)}.Params{count_param} ...
            = param;
    case 6
        stack.layers{idx(1)}.layers{idx(2)}.layers{idx(3)}.layers{idx(4)}.layers{idx(5)}.layers{idx(6)}.Params{count_param} ...
            = param;
    case 7
        stack.layers{idx(1)}.layers{idx(2)}.layers{idx(3)}.layers{idx(4)}.layers{idx(5)}.layers{idx(6)}.layers{idx(7)}.Params{count_param} ...
            = param;
otherwise
        error('the depth of layer is bigger than 7, NEED to adapt');

%% 

end
