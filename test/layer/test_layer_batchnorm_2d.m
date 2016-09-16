function test_layer_batchnorm_2d
% to check the gradiant of bias
% lichao, 20160914

%% parameters
p.le_tol = 1e-6;
p.ge_tol = 1e-11;
p.epsilon = 7e-5;
p.iterations = 1;
p.verbose = 'false';
p.add_linear = 'false';

inmap_size = [4  4];
inmaps_num = 2;
outmaps_num = 10;
% outmaps2_num = 4;

input_dim = prod(inmap_size)*inmaps_num;
mini_batch = 10;
%% build layer 1
conv2d_conf.inmaps_num = inmaps_num;
conv2d_conf.outmaps_num = outmaps_num;
conv2d_conf.inmap_size = inmap_size;
conv2d_conf.kernelsize = 3;
conv2d_conf.is_same_size = 'false';
%
conv2d_model = convolution2d_set(conv2d_conf);
conv2d_model = convolution2d_initial(conv2d_model);
%
outmap_size = conv2d_model.outmap_size;
output_dim = prod(outmap_size)*outmaps_num;
%
biaslayer_conf.indim = output_dim;
biaslayer_conf.inmaps_num = outmaps_num;
biaslayer_conf.inmap_size = outmap_size;
biaslayer_conf.axis_to_norm = 2;

biaslayer_model = batchnorm_set(biaslayer_conf);
biaslayer_model = batchnorm_initial(biaslayer_model);

%% input and labels

x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);

%% check grand 1
stack = stack_build({conv2d_model, biaslayer_model});
disp('check gradiant 2d bn');
check_grad_with_dummy(stack, x, y, p);


