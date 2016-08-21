function ce_cost_model = ce_cost_backward(ce_cost_model, y)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
h = ce_cost_model.h;

%% forward
% sross entropy
m = size(y,2); % number of sample
delta_in = h - y;
costv = -sum(sum(y .* log(h))) / m;

%% output and record
ce_cost_model.delta = delta_in;
ce_cost_model.costv = costv;


end
