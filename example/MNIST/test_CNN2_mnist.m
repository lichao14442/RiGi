function test_CNN2_mnist
% the main driver of CNN train and test in the datebase MNIST   
% lichao, 20170717

%% (1) load data
load data/mnist_uint8;
% train_x = single(reshape(train_x',28,28,60000))/255;
% test_x = single(reshape(test_x',28,28,10000))/255;
train_x = single(train_x'/255);
test_x = single(test_x'/255);
train_y = single(train_y');
test_y = single(test_y');

%% (2) CNN network config
% ex1 Train a 6c-2s-12c-2s Convolutional neural network
% modeldir = '';
% input data structure
high = 28;
wide = 28;
orimap_size = [high wide];
assert(size(train_x,1) == high * wide);
outdim = size(train_y, 1);
% outdim_full = 100;
rng(0)
batchsize = 100;
% nnet
nnet_conf.phase = 'train';
nnet_conf.order = 'whcn';

order = 'whcn';
bn_flag = 'true';
nonlinearity = 'sigmoid';
nnet_conf = {
    struct('type', 'input','name','input','inmaps_num',1,'inmap_size', orimap_size,...
        'indim',prod(orimap_size), 'batch_size',batchsize) %input layer
 
    struct('type', 'conv2dpack', 'name','conv1','outmaps_num', 6, 'kernelsize', 5,...
            'is_same_size', 'false', 'nonlinearity',nonlinearity,'batch_normalized',bn_flag,...
            'order', order) %convolution layer

    struct('type', 'pool2d', 'name','pool2x2-1','scale', 2,'method','average') %subsampling layer

    struct('type', 'conv2dpack', 'name','conv2','outmaps_num', 12, 'kernelsize', 5,...
            'is_same_size', 'false', 'nonlinearity',nonlinearity,'batch_normalized',bn_flag,...
            'order', order) %convolution layer

    struct('type', 'pool2d', 'name','pool2x2-2','scale', 2,'method','average') %subsampling layer

    struct('type', 'cePack','name','ce','outdim', outdim,'batch_normalized',bn_flag) 
};
%

%% (2.5) initial CNN network
cnn = nnet_setup(nnet_conf);

%% (3£©set training option
% option: 
opts.batchsize = batchsize;
opts.numepochs = 1;
opts.learningrate = 1;
opts.optimizer = 'moment';
opts.alpha = 0.9; % used when opts.optimizer == 'moment'
opts.verbose = 'false';

%% (3.5£©set training option
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
nnet_conf.phase = 'test';
[err_rate, bad] = nnet_test(cnn, opts, test_x, test_y);

disp(['the error rate in test set is ',num2str(err_rate)])
assert(err_rate<0.1, 'Too big error');
