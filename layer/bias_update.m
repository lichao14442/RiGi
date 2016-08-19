function bias_model = bias_update(bias_model,ops)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
% h = bias_model.h;
% x = bias_model.x;
% delta = bias_model.delta;
learningrate = ops.learningrate;
b = bias_model.b;
db = bias_model.db;

%% backward error
b = b - learningrate * db;

%% output and record
bias_model.b = b;

end




