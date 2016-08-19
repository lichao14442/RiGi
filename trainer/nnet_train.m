function nnet = nnet_train(nnet, opts, x, y)
% used to train nnet work
% first used to CNN network
% lichao 20160717
% data saved in 2d format, high(frequecy) first
% lichao 20160725

%% params of train 
batchsize = opts.batchsize;
numepochs = opts.numepochs;

%% training
num_sample = size(x, 2);
num_batches = floor(num_sample / batchsize);
disp(['  ** Using ' num2str(num_batches) ' batch per epotch to train model **']);
nnet.rL = [];
for i = 1 : numepochs
    disp(['--epoch ' num2str(i) '/' num2str(numepochs)]);
    tic;
    idx_shuffer = randperm(num_sample);
    for idx_batch = 1 : num_batches
        batch_x = x(:, idx_shuffer((idx_batch-1) * batchsize+1 : idx_batch * batchsize));
        batch_y = y(:, idx_shuffer((idx_batch-1) * batchsize+1 : idx_batch * batchsize));
        % forward
        nnet = nnet_forward(nnet, batch_x);
        % backward
        nnet = nnet_backward(nnet, opts, batch_y);
        % update
        nnet = nnet_update(nnet, opts);
        %
        if isempty(nnet.rL)
            nnet.rL(1) = nnet.costv;
            nnet.rL_smooth(1) = nnet.costv;
        elseif length( nnet.rL) == 1
            nnet.rL(end + 1) = nnet.costv;
            nnet.rL_smooth(end + 1) = nnet.costv;
        else
            nnet.rL(end + 1) = nnet.costv;
            nnet.rL_smooth(end + 1) = 0.95 * nnet.rL_smooth(end) + 0.05 * nnet.costv;
        end
    end
    toc;
end
    
end
