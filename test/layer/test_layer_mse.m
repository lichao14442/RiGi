function test_layer_mse
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
y = randn(output_dim, mini_batch);

%% build layer
mse_cost_conf.indim = input_dim;
mse_cost_model = mse_cost_set(mse_cost_conf);
mse_cost_model = mse_cost_initial(mse_cost_model);

%% check grand
check_grad_with_dummy(mse_cost_model, x, y);



