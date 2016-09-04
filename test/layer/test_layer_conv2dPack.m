function test_layer_conv2dPack
% to check the gradiant of bias
% lichao, 20160830

%% parameters
p.le_tol = 1e-6;
p.ge_tol = 1e-11;
p.epsilon = 7e-5;
p.iterations = 1;
p.verbose = 'true';
p.add_linear = 'false';

inmap_size = [5  5];
inmaps_num = 2;
outmaps_num = 8;
outmaps2_num = 16;

input_dim = prod(inmap_size)*inmaps_num;
mini_batch = 10;

nonlinearity = 'sigmoid';
batch_normalized = 'false';
is_same_size = 'false';
%% build layer 1
conv2dpack_conf.inmaps_num = inmaps_num;
conv2dpack_conf.outmaps_num = outmaps_num;
conv2dpack_conf.inmap_size = inmap_size;
conv2dpack_conf.kernelsize = 3;
conv2dpack_conf.nonlinearity = nonlinearity;
conv2dpack_conf.batch_normalized = batch_normalized;
conv2dpack_conf.is_same_size = is_same_size;
conv2dpack_conf.name = 'conv2dpack-1';
%
conv2dpack_model = conv2dPackage_set(conv2dpack_conf);
conv2dpack_model = conv2dPackage_initial(conv2dpack_model);

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
outmap_size = conv2dpack_model.outmap_size;
output_dim = prod(outmap_size)*outmaps_num;
x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);

%% check grand 1
disp('check gradiant 1 level convolution');
check_grad_with_dummy(conv2dpack_model, x, y, p);

%% 
conv2dpack2_conf.inmaps_num = outmaps_num;
conv2dpack2_conf.outmaps_num = outmaps2_num;
conv2dpack2_conf.inmap_size = outmap_size;
conv2dpack2_conf.kernelsize = 3;
conv2dpack2_conf.nonlinearity = nonlinearity;
conv2dpack2_conf.batch_normalized = batch_normalized;
conv2dpack2_conf.is_same_size = is_same_size;
%
conv2dpack2_model = conv2dPackage_set(conv2dpack2_conf);
conv2dpack2_model = conv2dPackage_initial(conv2dpack2_model);
%
outmap2_size = conv2dpack2_model.outmap_size;
output2_dim = prod(outmap2_size);
y2 = randn(output2_dim*outmaps2_num, mini_batch);

stack = stack_build({conv2dpack_model, conv2dpack2_model});
disp('check gradiant 2 level convolution');
check_grad_with_dummy(stack, x, y2, p);



