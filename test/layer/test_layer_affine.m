function test_layer_affine
% to check the gradiant of affine
% lichao, 20160831

%% parameters
input_dim = 10;
output_dim = 5;
mini_batch = 20;

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);

% for bn_flag = {'true','false'}
for bn_flag = {'true'}
    %% build layer
    affine_conf.indim = input_dim;
    affine_conf.outdim = output_dim;
    affine_conf.batch_normalized = bn_flag{1};
    %
    affine_model = affine_set(affine_conf);
    affine_model = affine_initial(affine_model);

    %% check grand
    check_grad_with_dummy(affine_model, x, y);

end

