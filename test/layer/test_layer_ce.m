function test_layer_ce
% to check the gradiant of bias
% lichao, 20160830

%% parameters
input_dim = 10;
output_dim = input_dim;
mini_batch = 20;

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
x = randn(input_dim, mini_batch);


% Make one-hot vectors (Cross Entropy)
y_onehot = zeros(input_dim, mini_batch);

for i = 1: mini_batch
    idx = randi(output_dim);
    y_onehot(idx, i) = 1.0;
end

%% build layer
nonlinear_conf.indim = output_dim;
nonlinear_conf.nonlinearity = 'softmax';
nonlinear_model = nonlinear_set(nonlinear_conf);
nonlinear_model = nonlinear_initial(nonlinear_model);

ce_cost_conf.indim = output_dim;
ce_cost_model = ce_cost_set(ce_cost_conf);
ce_cost_model = ce_cost_initial(ce_cost_model);

nnet.layers = {nonlinear_model,ce_cost_model};
nnet.layer_num = length(nnet.layers);
nnet.name = 'softmax+ce';
nnet.class = 'stack';
nnet.is_cost = 'true';
%% check grand
check_grad_with_dummy(nnet, x, y_onehot);



