function bias_model = bias_forward(bias_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
b = bias_model.b;

%% process
%  feedforward into output perceptrons
h = x + repmat(b, 1, size(x, 2));

%% output and record
bias_model.x = x;
bias_model.h = h;

end
