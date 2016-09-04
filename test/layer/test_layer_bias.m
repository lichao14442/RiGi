function test_layer_bias
% to check the gradiant of bias
% lichao, 20160830

%% parameters
p.le_tol = 1e-11;
p.ge_tol = 1e-16;
p.epsilon = 1e-7;
p.iterations = 1;
p.verbose = 'true';
p.add_linear = 'true';

input_dim = 10;
output_dim = input_dim;
mini_batch = 20;

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);

%% build layer
bias_conf.indim = output_dim;
bias_conf.name = 'bias';
bias_model = bias_set(bias_conf);
bias_model = bias_initial(bias_model);

%% check grand
check_grad_with_dummy(bias_model, x, y,p);



