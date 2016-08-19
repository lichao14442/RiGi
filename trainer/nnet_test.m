function [err_rate, bad] = nnet_test(nnet,ops, x, y)
%used to test the model 
% first used in CNN
%input:
%       nnet: model
%       x: in
%       y: label
%output:
%       err_rate: error
%       bad: badcase, where h ~= y
%
% lichao, 20160717

%% params of test 
batchsize = ops.batchsize;

%%  feedforward
% nnet = nnet_forward(nnet, x);
num_sample = size(x, 2);
num_batches = floor(num_sample / batchsize);
disp(['  ** Using ' num2str(num_batches) ' batch per epotch to test model **']);
h = single(zeros(size(y)));
tic;
for idx_batch = 1 : num_batches
    idx_sample = (idx_batch-1) * batchsize+1 : idx_batch * batchsize;
    batch_x = x(:,idx_sample);
    % forward
    nnet = nnet_set_testmode(nnet);
    nnet = nnet_forward(nnet, batch_x);
    h(:,idx_sample) = nnet.h;
end
toc;

%% static performance
[~, h_idx] = max(h);
[~, label] = max(y);
bad = find(h_idx ~= label);

err_rate = numel(bad) / size(y, 2);
end
