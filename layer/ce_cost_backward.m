function ce_cost_model = ce_cost_backward(ce_cost_model, y)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
h = ce_cost_model.h;
mini_batch = size(h,2);
%% backward
log_h = log(h);
h_x_y = log_h.* y;

% Temporarily fix to 2D reduce to work around majel bug
sum_all = sum(h_x_y(:));
costv = -sum_all/mini_batch;

% grad = h - y
delta_in =  (h - y)/mini_batch;

% % sross entropy
% m = size(y,2); % number of sample
% delta_in = (h - y)/mini_batch;
% costv = -sum(sum(y .* log(h))) / m;

%% output and record
ce_cost_model.delta = delta_in;
ce_cost_model.costv = costv;


end
