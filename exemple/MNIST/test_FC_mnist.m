function test_FC_mnist
% the main driver of CNN train and test in the datebase MNIST
% lichao, 20170717

%% (1) load data
load ../../testdata/mat_inout/mnist_uint8;
% train_x = single(reshape(train_x',28,28,60000))/255;
% test_x = single(reshape(test_x',28,28,10000))/255;
train_x = single(train_x'/255);
test_x = single(test_x'/255);
train_y = single(train_y');
test_y = single(test_y');

%% (2) initial CNN network
% ex1 Train a 6c-2s-12c-2s Convolutional neural network
% modeldir = '';
high = 28;
wide = 28;
orimap_size = [high wide];
assert(size(train_x,1) == high * wide);
outdim = size(train_y, 1);
% outdim_full = 100;
rng(0)
batchsize = 100;
order = 'wchn';
nnet_conf = {
    struct('type', 'input','name','input','inmap_size', orimap_size,'indim',prod(orimap_size), 'batch_size',batchsize) %input layer
 
    struct('type', 'fullconnect','name','full1', 'outdim', 128, 'nonlinearity','sigmoid') %subsampling layer
    struct('type', 'batchnorm', 'name','bn1') % batch norm
    struct('type', 'fullconnect','name','full2', 'outdim', 128, 'nonlinearity','sigmoid') %subsampling layer
    struct('type', 'batchnorm', 'name','bn2') % batch norm
    struct('type', 'fullconnect','name','full3', 'outdim', outdim, 'nonlinearity','sigmoid') %subsampling layer
    %     struct('type', 'fullconnect','name','full2', 'outdim', outdim, 'nonlinearity', 'sigmoid') %subsampling layer
%     struct('type', 'nonlinear','name','softmax', 'nonlinearity', 'softmax') %subsampling layer
    struct('type', 'mse-cost','name','mse') %subsampling layer
};
%
cnn = nnet_setup(nnet_conf);

%% (3£©training
% option: 
opts.batchsize = batchsize;
opts.numepochs = 1;
opts.learningrate = 1;
opts.optimizer = 'moment';
opts.alpha = 0.9; % used when opts.optimizer == 'moment'

%will run 1 epoch in about 100 second and get around 6% error. 
%With 100 epochs you'll get around 1% error
cnn = nnet_train(cnn, opts, train_x, train_y);

%plot mean squared error
idx = 1: length(cnn.rL);
figure; plot(idx, cnn.rL, 'b', idx, cnn.rL_smooth, 'r');
box on; grid on;
legend('record-cost','smooth-cost');
disp(['the mincost is ', num2str(min(cnn.rL))])

%% (4) testing
[err_rate, bad] = nnet_test(cnn, opts, test_x, test_y);

disp(['the error rate in test set is ',num2str(err_rate)])
assert(err_rate<0.1, 'Too big error');
