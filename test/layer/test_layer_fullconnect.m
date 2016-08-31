function test_layer_fullconnect
% to check the gradiant of affine
% lichao, 20160831

%% parameters
input_dim = 10;
output_dim = 5;
mini_batch = 20;

%% input and labels
x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);

%% build layer
full_conf.indim = input_dim;
full_conf.outdim = output_dim;
full_conf.batch_normalized = 'false';
%
full_model = fullconnect_set(full_conf);
full_model = fullconnect_initial(full_model);

%% check grand
check_grad_with_dummy(full_model, x, y);



