function bias_model = bias_backward(bias_model,ops, delta)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
param_dim = bias_model.param_dim;
outmap_size = bias_model.outmap_size;
outmaps_num = bias_model.outmaps_num;
axis_to_norm = bias_model.axis_to_norm;

optimizer = ops.optimizer;
% optimizer = 'sgd';
%% backward error

%% comput gradient
% (1) reshape
delta_2d = delta;
[indim, batch_size] = size(delta);
if axis_to_norm == 0
    delta_reshaped = delta;
%     batch_size_new = batch_size;
else % ONLY support 2 || 3
    batch_size_new = indim/param_dim*batch_size;
    delta_reshaped = zeros(param_dim, batch_size_new);
    if axis_to_norm == 2
        allmap_size = [outmap_size(1),outmaps_num,outmap_size(2),batch_size];
        delta = reshape(delta,allmap_size);
        for i = 1: param_dim
            delta_reshaped(i,:) = reshape(delta(:,i,:,:),1,batch_size_new);
        end
    elseif axis_to_norm == 3
        allmap_size = [outmap_size(1),outmap_size(2),outmaps_num,batch_size];
        delta = reshape(delta,allmap_size);
        for i = 1: param_dim
            delta_reshaped(i,:) = reshape(delta(:,:,i,:),1,batch_size_new);
        end
    end
end

% (2) process
db = sum(delta_reshaped, 2);
if strcmp(optimizer,'sgd')
    bias_model.dParams{1} = db;
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    bias_model.dParams{1} = alpha .* bias_model.dParams{1} + db;
end

%% output and record
bias_model.delta = delta_2d;
end




