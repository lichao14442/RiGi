function test_layer_cePack
% to check the gradiant of bias
% lichao, 20160830

%% parameters
p.le_tol = 1e-11;
p.ge_tol = 1e-16;
p.epsilon = 1e-7;
p.iterations = 1;
p.verbose = 'true';
p.add_linear = 'flase';

input_dim = 10;
output_dim = 5;
mini_batch = 20;

%% input and labels
x = randn(input_dim, mini_batch);

% Make one-hot vectors (Cross Entropy)
y_onehot = zeros(output_dim, mini_batch);

for i = 1: mini_batch
    idx = randi(output_dim);
    y_onehot(idx, i) = 1.0;
end

%% build layer

cePack_conf.indim = input_dim;
cePack_conf.outdim = output_dim;
cePackage_model = cePackage_set(cePack_conf);
cePackage_model = cePackage_initial(cePackage_model);

%% check grand
check_grad_with_dummy(cePackage_model, x, y_onehot,p);



