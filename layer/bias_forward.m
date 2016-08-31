function bias_model = bias_forward(bias_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
param_dim = bias_model.param_dim;
outmap_size = bias_model.outmap_size;
outmaps_num = bias_model.outmaps_num;
axis_to_norm = bias_model.axis_to_norm;
%
b = bias_model.Params{1};

%% process
%  feedforward into output perceptrons
%(1) reshape
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

% (2) process
h_reshaped = bsxfun(@plus, x_reshaped, b);

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
bias_model.x = x_ori;
bias_model.h = h;

end
