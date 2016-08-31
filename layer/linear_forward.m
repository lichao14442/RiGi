function linear_model = linear_forward(linear_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
W = linear_model.Params{1};

%% process
%  feedforward into output perceptrons
h = W * x;

%% output and record
linear_model.x = x;
linear_model.h = h;

end
