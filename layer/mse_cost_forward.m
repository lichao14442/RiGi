function mse_cost_model = mse_cost_forward(mse_cost_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params


%% forward
h = x;

%% output and record
mse_cost_model.x = x;
mse_cost_model.h = h;

end
