function test_layer_linear
% to check the gradiant of bias
% lichao, 20160830

%% parameters
input_dim = 10;
output_dim = 15;
mini_batch = 20;

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);

%% build layer
linear_conf.indim = input_dim;
linear_conf.outdim = output_dim;
linear_model = linear_set(linear_conf);
linear_model = linear_initial(linear_model);

%% check grand
check_grad_with_dummy(linear_model, x, y);



