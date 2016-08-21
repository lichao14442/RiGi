function batchnorm_model = batchnorm_forward(batchnorm_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
param_dim = batchnorm_model.param_dim;
outmap_size = batchnorm_model.outmap_size;
outmaps_num = batchnorm_model.outmaps_num;
axis_to_norm = batchnorm_model.axis_to_norm;
gamma = batchnorm_model.gamma;
beta = batchnorm_model.beta;
running_mean = batchnorm_model.running_mean;
running_var = batchnorm_model.running_var;
running_samples = batchnorm_model.running_samples;
running_iters = batchnorm_model.running_iters;
eval_mean = batchnorm_model.eval_mean;
eval_var = batchnorm_model.eval_var;
test_mode = batchnorm_model.test_mode;
bn_eval_stats = batchnorm_model.bn_eval_stats;
% outdim = batchnorm_model.outdim;

%% process
%  feedforward into output perceptrons
% (1) reshape
x_ori = x;
[indim, batch_size] = size(x);
if axis_to_norm == 0
    x_reshaped = x;
else % ONLY support 2 || 3
    batch_size_new = indim/param_dim*batch_size;
    x_reshaped = zeros(param_dim, batch_size_new);
    if axis_to_norm == 2
        allmap_size = [outmap_size(1),outmaps_num,outmap_size(2),batch_size];
        x = reshape(x,allmap_size);
        for i = 1: param_dim
            x_reshaped(i,:) = reshape(x(:,i,:,:),1,batch_size_new);
        end
    elseif axis_to_norm == 3
        allmap_size = [outmap_size(1),outmap_size(2),outmaps_num,batch_size];
        x = reshape(x,allmap_size);
        for i = 1: param_dim
            x_reshaped(i,:) = reshape(x(:,:,i,:),1,batch_size_new);
        end
    end
end

% (2) batch normalization
if strcmp(test_mode, 'true') % TEST MODE
    if sum(eval_var) ~= 0
        disp('using eval mean and var to test');
        x_hat = normalize_batch(x_reshaped, eval_mean, eval_var);
    elseif sum(running_var) ~= 0
        disp('using running mean and var to test');
        x_hat = normalize_batch(x_reshaped, running_mean, running_var);
    else
        error('the statistic are NOT aviable');
    end
else % not TEST MODE
    [batch_mean, batch_var, running_mean, running_var, running_samples] = ...
        comput_mean_var(x_reshaped,running_samples,running_mean, running_var);
    x_hat = normalize_batch(x_reshaped, batch_mean, batch_var);
%     disp ([ ' the differ mean is ', num2str(sum((running_mean - batch_mean).^2))]);
%     disp([' the differ var is ', num2str(sum((running_var - batch_var).^2))]);
%    disp ( [' norm of batchmean is ', num2str( norm(batch_mean,2))]);
    %
    running_iters = running_iters + 1;
    if running_iters >= bn_eval_stats
        batchnorm_model.eval_mean = running_mean;
        batchnorm_model.eval_var = running_var;
        running_mean = single(zeros(param_dim, 1));
        running_var = single(zeros(param_dim, 1));
        running_samples = 0;
        running_iters = 0;
    end
    batchnorm_model.batch_mean = batch_mean;
    batchnorm_model.batch_var = batch_var;
end

h_reshaped = bsxfun(@times,x_hat,gamma);
h_reshaped = bsxfun(@plus, h_reshaped, beta);

%(3) reshape back
if axis_to_norm == 0
    h = h_reshaped;
else % ONLY support 2 || 3
    h = zeros(allmap_size);
    if axis_to_norm == 2
        map_size = [outmap_size(1),1,outmap_size(2),batch_size];
        for i = 1: param_dim
            h(:,i,:,:) = reshape(h_reshaped(i,:),map_size);
        end
    elseif axis_to_norm == 3
        map_size = [outmap_size(1),outmap_size(2),1,batch_size];
        for i = 1: param_dim
            h(:,:,i,:) = reshape(h_reshaped(i,:),map_size);
        end
    end
    h = reshape(h,indim, batch_size);
end

%% output and record
batchnorm_model.x = x_ori;
batchnorm_model.h = h;
%
batchnorm_model.running_mean = running_mean;
batchnorm_model.running_var = running_var;
batchnorm_model.running_samples = running_samples;
batchnorm_model.running_iters = running_iters;

end

function h = normalize_batch(x, mean_loc, var_loc)
    h = bsxfun(@minus, x, mean_loc);
    std = sqrt(var_loc + eps);
    h = bsxfun(@rdivide, h, std);
end

function  [batch_mean, batch_var, running_mean, running_var, running_samples] = ...
        comput_mean_var(x, running_samples, running_mean, running_var)
    batch_size = size(x,2);
    total_size = batch_size + running_samples;
    prev_ratio = running_samples / total_size;
    curr_ratio = batch_size / total_size;
    harmonic_ratio = batch_size * running_samples / total_size / total_size;
    %
    batch_mean = mean(x,2);
    delta = running_mean - batch_mean;
    running_mean = prev_ratio * delta + batch_mean;
    %
    x_hat = bsxfun(@minus, x, batch_mean);
    batch_var = mean(x_hat.^2, 2);
    running_var = curr_ratio * batch_var + prev_ratio * running_var + harmonic_ratio * delta.^2;
    %
    running_samples = total_size;
end
    
    


