function full_model = fullLinear_backward(full_model,ops, delta)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
h = full_model.h;
x = full_model.x;
% delta = full_model.delta;
optimizer = ops.optimizer;
W = full_model.W;
dW = full_model.dW;
db = full_model.db;
%% backward error
d_h = delta .* (h .* (1 - h));                 %  output delta
delta_in = (W' * d_h);                   %  feature vector delta

%% comput gradient
dW = d_h * x' / size(d_h, 2);
db = mean(d_h, 2);
% dW = d_h * x' ;
% db = sum(d_h, 2);
if strcmp(optimizer,'sgd')
    full_model.db = db;
    full_model.dW = dW;
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    full_model.dW = alpha .* full_model.dW + dW;
    full_model.db = alpha .* full_model.db + db;
end

%% output and record
full_model.delta = delta_in;

end




