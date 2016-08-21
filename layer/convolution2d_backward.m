function conv2d_model = convolution2d_backward(conv2d_model, ops, delta)
% forward of convolution2d layer
% model: 
% ops : option
% lichao , 20160717

%% params
inmaps_num = conv2d_model.inmaps_num;
outmaps_num = conv2d_model.outmaps_num;
inmap_size = conv2d_model.inmap_size;
outmap_size = conv2d_model.outmap_size;
kernelsize = conv2d_model.kernelsize;
is_same_size = conv2d_model.is_same_size;
order = conv2d_model.order;
% h = conv2d_model.h;
x = conv2d_model.x;
% delta = conv2d_model.delta;
optimizer = ops.optimizer;
k = conv2d_model.k;
% b = conv2d_model.b;
% dk = conv2d_model.dk;
% db = conv2d_model.db;
%
%% backward delta
%  !!below can probably be handled by insane matrix operations
num_sample = size(delta,2);
switch (order)
    case 'whcn'
        size_x = [inmap_size, inmaps_num, num_sample];
        size_h = [outmap_size, outmaps_num, num_sample];
    case 'wchn'
        size_x = [inmap_size(1), inmaps_num, inmap_size(2), num_sample];
        size_h = [outmap_size(1), outmaps_num, outmap_size(2), num_sample];
    otherwise
        error('the order NOT support right new!');
end
x = reshape(x, size_x);
delta = reshape(delta, size_h);
delta_in = single(zeros(size_x));

batch_inonemap_size = [inmap_size, num_sample];
batch_outonemap_size = [outmap_size, num_sample];

if strcmp(is_same_size,'false')
    ops_conv = 'full';
    batch_inonemap_zerosend_size = batch_inonemap_size;
else
    ops_conv = 'same';
    batch_inonemap_zerosend_size = batch_inonemap_size + [kernelsize-1 kernelsize-1 0];
    dimrange_w = (kernelsize-1)/2 + 1 :batch_inonemap_zerosend_size(1) - (kernelsize-1)/2;
    dimrange_h = (kernelsize-1)/2 + 1 :batch_inonemap_zerosend_size(2) - (kernelsize-1)/2;
end
%
for i = 1 : inmaps_num
    z = single(zeros(batch_inonemap_size));
    for j = 1 :outmaps_num
        switch (order)
            case 'whcn'
                delta_loc = delta(:,:,j,:);
            case 'wchn'
                delta_loc = delta(:,j,:,:);
        end
        delta_3d = reshape(delta_loc, batch_outonemap_size);
        z = z + convn(delta_3d, rot180(k(:,:,i,j)), ops_conv);
    end
    switch (order)
        case 'whcn'
            delta_in(:,:,i,:) = z;
        case 'wchn'
            delta_in(:,i,:,:) = z;
    end
end
size_x_out = [prod([inmap_size, inmaps_num]), num_sample];
delta_in = reshape(delta_in,size_x_out);

%% compute giadient
dk = single(zeros(kernelsize, kernelsize, inmaps_num, outmaps_num));
% db = single(zeros(1, outmaps_num));
for j = 1 : outmaps_num
    switch (order)
        case 'whcn'
            delta_loc = delta(:,:,j,:);
        case 'wchn'
            delta_loc = delta(:,j,:,:);
    end
    delta_3d = reshape(delta_loc, batch_outonemap_size);
    for i = 1 :inmaps_num
        switch (order)
            case 'whcn'
                x_loc = x(:,:,i,:);
            case 'wchn'
                x_loc = x(:,i,:,:);
        end
        x_loc = reshape(x_loc, batch_inonemap_size);
        if strcmp(is_same_size,'true') % 吧 x 四周变大
            x_zerosend = single(zeros(batch_inonemap_zerosend_size));
            x_zerosend(dimrange_w, dimrange_h, :) = x_loc;
        else
            x_zerosend = reshape(x_loc, batch_inonemap_zerosend_size);
        end
        dk(:,:,i,j) = convn(flipall(x_zerosend),delta_3d, 'valid') / size(delta_3d, 3);
    end
%     db(j) = sum(delta_3d(:)) / size(delta_3d, 3);
end
if strcmp(optimizer,'sgd')
%     conv2d_model.db = db;
    conv2d_model.dk = dk;
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    conv2d_model.dk = alpha .* conv2d_model.dk + dk;
%     conv2d_model.db = alpha .* conv2d_model.db + db;
end

%% output and record
conv2d_model.delta = delta_in;

end
function X = rot180(X)
    X = flipdim(flipdim(X, 1), 2);
end