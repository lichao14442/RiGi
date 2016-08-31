function [cost, nnet] = cost_and_gradient(nnet, x, y) 
% this function is used to compute the cost and gradiant
% lichao, 20160830
%

%% Forward 
nnet = nnet_forward(nnet, x);

%% Backward 
opts.optimizer = 'sgd';
nnet = nnet_backward(nnet, opts, y);

%% output
cost = nnet.costv;
