function batchnorm_model = batchnorm_backward(batchnorm_model,ops, delta)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160725
%

%% params
param_dim = batchnorm_model.param_dim;
outmap_size = batchnorm_model.outmap_size;
outmaps_num = batchnorm_model.outmaps_num;
axis_to_norm = batchnorm_model.axis_to_norm;
x = batchnorm_model.x;
% h = batchnorm_model.h;
% running_mean = batchnorm_model.running_mean;
% running_var = batchnorm_model.running_var;
batch_mean = batchnorm_model.batch_mean;
batch_var = batchnorm_model.batch_var;
% eval_mean = batchnorm_model.eval_mean;
% eval_var = batchnorm_model.eval_var;
%
optimizer = ops.optimizer;

gamma = batchnorm_model.Params{1};
%% backward error
% (1) reshape
[indim, batch_size] = size(x);
if axis_to_norm == 0
    x_reshaped = x;
    delta_reshaped = delta;
    batch_size_new = batch_size;
else % ONLY support 2 || 3
    batch_size_new = indim/param_dim*batch_size;
    x_reshaped = zeros(param_dim, batch_size_new);
    delta_reshaped = zeros(param_dim, batch_size_new);
    if axis_to_norm == 2
        allmap_size = [outmap_size(1),outmaps_num,outmap_size(2),batch_size];
        x = reshape(x,allmap_size);
        delta = reshape(delta,allmap_size);
        for i = 1: param_dim
            x_reshaped(i,:) = reshape(x(:,i,:,:),1,batch_size_new);
            delta_reshaped(i,:) = reshape(delta(:,i,:,:),1,batch_size_new);
        end
    elseif axis_to_norm == 3
        allmap_size = [outmap_size(1),outmap_size(2),outmaps_num,batch_size];
        x = reshape(x,allmap_size);
        delta = reshape(delta,allmap_size);
        for i = 1: param_dim
            x_reshaped(i,:) = reshape(x(:,:,i,:),1,batch_size_new);
            delta_reshaped(i,:) = reshape(delta(:,:,i,:),1,batch_size_new);
        end
    end
end
% (2) process
x_hat = normalize_batch(x_reshaped, batch_mean, batch_var);
d_x_hat = bsxfun(@times, delta_reshaped, gamma);
std = sqrt(batch_var+eps);
%
tmp = bsxfun(@minus, x_reshaped, batch_mean);
d_var =  sum( d_x_hat .*tmp , 2);
d_var = -0.5 * d_var ./(std.^3);
%
d_mean = -1 * sum(d_x_hat,2) ./std ;
d_mean = d_mean - 2* mean(tmp, 2) .* d_var;
% 
delta_in_reshaped = bsxfun(@rdivide, d_x_hat, std) + 2/batch_size_new .*bsxfun(@times, tmp, d_var);
delta_in_reshaped = bsxfun(@plus, delta_in_reshaped, d_mean/batch_size_new);

% delta_in_reshaped = bsxfun(@rdivide, d_x_hat, std) + 2 .*bsxfun(@times, tmp, d_var);
% delta_in_reshaped = bsxfun(@plus, delta_in_reshaped, d_mean);

%(3)reshape back
if axis_to_norm == 0
    delta_in = delta_in_reshaped;
else % ONLY support 2 || 3
    delta_in = zeros(allmap_size);
    if axis_to_norm == 2
        map_size = [outmap_size(1),1,outmap_size(2),batch_size];
        for i = 1: param_dim
            delta_in(:,i,:,:) = reshape(delta_in_reshaped(i,:),map_size);
        end
    elseif axis_to_norm == 3
        map_size = [outmap_size(1),outmap_size(2),1,batch_size];
        for i = 1: param_dim
            delta_in(:,:,i,:) = reshape(delta_in_reshaped(i,:),map_size);
        end
    end
    delta_in = reshape(delta_in,indim, batch_size);
end

%% comput gradient
if sum(batch_var) == 0
    error('the var is zeros');
end

x_hat = x_hat .* delta_reshaped;
dgamma = sum(x_hat,2);
dbeta = sum(delta_reshaped,2);
%
if strcmp(optimizer,'sgd')
    batchnorm_model.dParams{1} = dgamma;
    batchnorm_model.dParams{2} = dbeta;
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    batchnorm_model.dParams{1} = alpha .* batchnorm_model.dParams{1} + dgamma;
    batchnorm_model.dParams{2} = alpha .* batchnorm_model.dParams{2} + dbeta;
end

%% output and record
batchnorm_model.delta = delta_in;

end

function h = normalize_batch(x, mean_loc, var_loc)
    h = bsxfun(@minus, x, mean_loc);
    std = sqrt(var_loc + eps);
    h = bsxfun(@rdivide, h, std);
end


