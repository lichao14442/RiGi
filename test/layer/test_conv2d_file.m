function test_conv2d_file

if nargin == 0
   filetype = 'npy';
   model_dir = ['../../testdata/',filetype,'_model/conv2d'];
   x_file = ['../../testdata/',filetype,'_model/conv2d/x.npy'];
   delta_file = ['../../testdata/',filetype,'_model/conv2d/delta.npy'];
   h_file = ['../../testdata/',filetype,'_model/conv2d/h_mt.npy'];
   deltain_file = ['../../testdata/',filetype,'_model/conv2d/delta_mt.npy'];
end

inmaps_num = 5;
outmaps_num = 10;
inmap_size = [6 11 ];
kernelsize = 3;

%% 初始化，模型
conv2d_conf.inmaps_num = inmaps_num;
conv2d_conf.outmaps_num = outmaps_num;
conv2d_conf.inmap_size = inmap_size;
conv2d_conf.kernelsize = kernelsize;
conv2d_conf.is_same_size = 'false';
conv2d_conf.order = 'wchn';

conv2d_model = convolution2d_set(conv2d_conf);
conv2d_model = convolution2d_initial(conv2d_model);
conv2d_model = convolution2d_load(model_dir,0,conv2d_model);
%

%% 生成input的随机数
%数据是行优先的，每一行是一帧
[x,~ , batch_size] = load_inputdata(x_file);
[delta, ~, ~] = load_inputdata(delta_file);

%% 前向打分
conv2d_model = convolution2d_forward(conv2d_model,x);
h = conv2d_model.h;
%
opts.batchsize = batch_size;
opts.learningrate = 1;
opts.optimizer = 'moment';
opts.alpha = 0.9; % used when opts.optimizer == 'moment'
% conv2d_model = convolution2d_backward(conv2d_model, opts, delta);
% delta_in = conv2d_model.delta;

%% 输出y
% TODO: save to txt file
%savedata(y,y_file);
save_outputdata(h, h_file);
% save_outputdata(delta_in, deltain_file);

