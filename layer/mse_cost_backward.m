function mse_cost_model = mse_cost_backward(mse_cost_model, y)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
h = mse_cost_model.h;

%% forward
% sross entropy
m = size(y,2); % number of sample
delta_in = h - y;
%  loss function
costv = 1/2* sum(delta_in(:) .^ 2) / m;

%% output and record
mse_cost_model.delta = delta_in;
mse_cost_model.costv = costv;


end
