function linear_model = linear_update(linear_model,ops)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
% h = linear_model.h;
% x = linear_model.x;
% delta = linear_model.delta;
learningrate = ops.learningrate;
W = linear_model.W;
dW = linear_model.dW;

%% backward error
W = W - learningrate * dW;

%% output and record
linear_model.W = W;

end




