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

% for bn_flag = {'true','false'}
for bn_flag = {'false'}
    %% build layer
    full_conf.indim = input_dim;
    full_conf.outdim = output_dim;
    full_conf.batch_normalized = bn_flag{1};
    %
    full_model = fullconnect_set(full_conf);
    full_model = fullconnect_initial(full_model);

    %% check grand
    disp('check gradiant 1 level full contection');
    check_grad_with_dummy(full_model, x, y);
    
    %% build layer 2-level
    full2_conf.indim = output_dim;
    full2_conf.outdim = output_dim;
    full2_conf.batch_normalized = bn_flag{1};
    %
    full2_model = fullconnect_set(full2_conf);
    full2_model = fullconnect_initial(full2_model);

    %% check grand
    stack = stack_build({full_model, full2_model});
    disp('check gradiant 2 level full contection');
    check_grad_with_dummy(stack, x, y);

end


