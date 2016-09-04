function ce_cost_model = ce_cost_forward(ce_cost_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params


%% forward
% softmax
h = exp(bsxfun(@minus, x, max(x,[],1)));
h = bsxfun(@rdivide, h, sum(h, 1)); 

%% output and record
ce_cost_model.x = x;
ce_cost_model.h = h;

end
