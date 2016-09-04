function test_layer_pooling2d
% to check the gradiant of bias
% lichao, 20160830

%% parameters
p.le_tol = 1e-11;
p.ge_tol = 1e-16;
p.epsilon = 1e-7;
p.iterations = 1;
p.verbose = 'true';
p.add_linear = 'true';

inmap_size = [4  4];
inmaps_num = 2;
outmaps_num = inmaps_num;
% outmaps2_num = 4;

input_dim = prod(inmap_size);
mini_batch = 10;

%% build layer 1
pool2d_conf.inmaps_num = inmaps_num;
pool2d_conf.inmap_size = inmap_size;
pool2d_conf.scale = 2;
pool2d_conf.method = 'max';
%
pool2d_model = pooling2d_set(pool2d_conf);
pool2d_model = pooling2d_initial(pool2d_model);

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
outmap_size = pool2d_model.outmap_size;
output_dim = prod(outmap_size);
x = randn(input_dim*inmaps_num, mini_batch);
y = randn(output_dim*outmaps_num, mini_batch);
%     auto y_cconv2d_1 = rand(make_ddim({static_cast<int>(p.filter_num_1 * same_conv_dim(p.input_freq, p.stride_freq) * same_conv_dim(p.input_time, p.stride_time)),
%                 static_cast<int>(mini_batch * 10)}), double(0));
%     auto y_cconv2d_2 = rand(make_ddim({p.filter_num_2 * static_cast<int>(same_conv_dim(p.input_freq, p.stride_freq) * same_conv_dim(p.input_time, p.stride_time)),
%                 static_cast<int>(10 * mini_batch)}), double(0));
%     auto y_cconv2d_fc = rand(make_ddim({output_dim, mini_batch * 10}), double(0));

%% check grand 1
disp('check gradiant pooling');
check_grad_with_dummy(pool2d_model, x, y, p);

