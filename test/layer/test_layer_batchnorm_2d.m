function test_layer_batchnorm_2d
% to check the gradiant of bias
% lichao, 20160830

%% parameters
inmap_size = [ 4, 6];
inmaps_num = 5;
outmaps_num = 7;

input_dim = prod(inmap_size);
output_dim = input_dim;
mini_batch = 20;

%% input and labels
% x = single(randn(input_dim, mini_batch));
% y = single(randn(output_dim, mini_batch));
x = randn(input_dim*inmaps_num, mini_batch);
y = randn(output_dim*outmaps_num, mini_batch);

%% build layer
outmap_size = conv2d_model.outmap_size;
outdim = outmaps_num * prod(outmap_size);
biaslayer_conf.indim = outdim;
biaslayer_conf.inmaps_num = outmaps_num;
biaslayer_conf.inmap_size = outmap_size;
biaslayer_conf.axis_to_norm = strfind(order,'c');
if strcmp(batch_normalized,'true')
    biaslayer_conf.name = [conv2dpack_model.name,'->batchnorm'];
    biaslayer_model = batchnorm_set(biaslayer_conf);
    biaslayer_model = batchnorm_initial(biaslayer_model);

end


batchnorm_conf.indim = output_dim;
%
batchnorm_model = batchnorm_set(batchnorm_conf);
batchnorm_model = batchnorm_initial(batchnorm_model);

%% check grand
check_grad_with_dummy(batchnorm_model, x, y);



