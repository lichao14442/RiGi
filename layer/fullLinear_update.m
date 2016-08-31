function full_model = fullLinear_update(full_model,ops)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
% h = full_model.h;
% x = full_model.x;
% delta = full_model.delta;
learningrate = ops.learningrate;
W = full_model.W;
b = full_model.b;
dW = full_model.dW;
db = full_model.db;
%% backward error
W = W - learningrate * dW;
b = b - learningrate * db;

%% output and record
full_model.W = W;
full_model.b = b;
end




