function linear_model = linear_backward(linear_model,ops, delta)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
h = linear_model.h;
x = linear_model.x;
% delta = linear_model.delta;
optimizer = ops.optimizer;
W = linear_model.W;

%% backward error
delta_in = (W' * delta);                   %  feature vector delta

%% comput gradient
dW = delta * x' / size(delta, 2);
% dW = delta * x';
if strcmp(optimizer,'sgd')
    linear_model.dW = dW;
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    linear_model.dW = alpha .* linear_model.dW + dW;
end

%% output and record
linear_model.delta = delta_in;


end




