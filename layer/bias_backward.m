function bias_model = bias_backward(bias_model,ops, delta)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
h = bias_model.h;
x = bias_model.x;
% delta = bias_model.delta;
optimizer = ops.optimizer;

%% backward error

%% comput gradient
db = mean(delta, 2);
if strcmp(optimizer,'sgd')
    bias_model.db = db;
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    bias_model.db = alpha .* bias_model.db + db;
end

%% output and record
bias_model.delta = delta;
end




