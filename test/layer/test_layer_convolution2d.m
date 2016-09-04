function test_layer_convolution2d
% to check the gradiant of bias
% lichao, 20160830

%% parameters
p.le_tol = 1e-6;
p.ge_tol = 1e-11;
p.epsilon = 7e-5;
p.iterations = 1;
p.verbose = 'true';
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

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
outmap_size = conv2d_model.outmap_size;
output_dim = prod(outmap_size)*outmaps_num;
x = randn(input_dim, mini_batch);
y = randn(output_dim, mini_batch);
%     auto y_cconv2d_1 = rand(make_ddim({static_cast<int>(p.filter_num_1 * same_conv_dim(p.input_freq, p.stride_freq) * same_conv_dim(p.input_time, p.stride_time)),
%                 static_cast<int>(mini_batch * 10)}), double(0));
%     auto y_cconv2d_2 = rand(make_ddim({p.filter_num_2 * static_cast<int>(same_conv_dim(p.input_freq, p.stride_freq) * same_conv_dim(p.input_time, p.stride_time)),
%                 static_cast<int>(10 * mini_batch)}), double(0));
%     auto y_cconv2d_fc = rand(make_ddim({output_dim, mini_batch * 10}), double(0));

%% check grand 1
disp('check gradiant 1 level convolution');
check_grad_with_dummy(conv2d_model, x, y, p);


% %% 
% conv2d2_conf.inmaps_num = outmaps_num;
% conv2d2_conf.outmaps_num = outmaps2_num;
% conv2d2_conf.inmap_size = outmap_size;
% conv2d2_conf.kernelsize = 1;
% conv2d2_conf.is_same_size = 'false';
% %
% conv2d2_model = convolution2d_set(conv2d2_conf);
% conv2d2_model = convolution2d_initial(conv2d2_model);
% %
% outmap2_size = conv2d2_model.outmap_size;
% output2_dim = prod(outmap2_size);
% y2 = randn(output2_dim*outmaps2_num, mini_batch);
% 
% stack = stack_build({conv2d_model, conv2d2_model});
% disp('check gradiant 2 level convolution');
% check_grad_with_dummy(stack, x, y2, p);










