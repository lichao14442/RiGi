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
batch_mean = batchnorm_model.batch_mean;
batch_var = batchnorm_model.batch_var;
gamma = batchnorm_model.Params{1};
%
go_ops_order = batchnorm_model.go_ops_order;
back_ops_order = batchnorm_model.back_ops_order;
%
optimizer = ops.optimizer;

%% backward error
%delta_ori = delta; % for debug
% (1) reshape
[indim, batch_size] = size(x);
if axis_to_norm == 0
    x_reshaped = x;
    delta_reshaped = delta;
    batch_size_new = batch_size;
else
    allmap_size = [outmap_size(1),outmap_size(2),outmaps_num,batch_size];
    batch_size_new = prod(allmap_size)/param_dim;
    x = reshape(x,allmap_size);
    x_orderd = permute(x,go_ops_order);
    x_reshaped = reshape(x_orderd,param_dim,batch_size_new);  
    delta = reshape(delta, allmap_size);
    delta_orderd = permute(delta, go_ops_order);
    delta_reshaped = reshape(delta_orderd, param_dim, batch_size_new);
    orderedmap_size = allmap_size(go_ops_order);
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

% (3)reshape back
if axis_to_norm == 0
    delta_in = delta_in_reshaped;
else
    delta_in = reshape(delta_in_reshaped,orderedmap_size);
    delta_in_ordered = permute(delta_in,back_ops_order);
    delta_in = reshape(delta_in_ordered,indim, batch_size);
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


