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
% h = conv2d_model.h;
x = conv2d_model.x;
% delta = conv2d_model.delta;
optimizer = ops.optimizer;
k = reshape(conv2d_model.Params{1},[kernelsize, kernelsize,inmaps_num, outmaps_num]);
%
%% backward delta
num_sample = size(delta,2);
size_x = [inmap_size, inmaps_num, num_sample];
size_h = [outmap_size, outmaps_num, num_sample];
x = reshape(x, size_x);
delta = reshape(delta, size_h);
delta_in = zeros(size_x);

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
    z = zeros(batch_inonemap_size);
    for j = 1 :outmaps_num
        delta_loc = delta(:,:,j,:);
        delta_3d = reshape(delta_loc, batch_outonemap_size);
        z = z + convn(delta_3d, rot180(k(:,:,i,j)), ops_conv);
    end
    delta_in(:,:,i,:) = z;
end
size_x_out = [prod([inmap_size, inmaps_num]), num_sample];
delta_in = reshape(delta_in,size_x_out);

%% compute giadient
dk = zeros(kernelsize, kernelsize, inmaps_num, outmaps_num);

for j = 1 : outmaps_num
    delta_loc = delta(:,:,j,:);
    delta_3d = reshape(delta_loc, batch_outonemap_size);
    for i = 1 :inmaps_num
        x_loc = x(:,:,i,:);
        x_loc = reshape(x_loc, batch_inonemap_size);
        if strcmp(is_same_size,'true') % �� x ���ܱ��
            x_zerosend = zeros(batch_inonemap_zerosend_size);
            x_zerosend(dimrange_w, dimrange_h, :) = x_loc;
        else
            x_zerosend = reshape(x_loc, batch_inonemap_zerosend_size);
        end
%         dk(:,:,i,j) = convn(flipall(x_zerosend),delta_3d, 'valid') / size(delta_3d, 3);
        dk(:,:,i,j) = convn(flipall(x_zerosend),delta_3d, 'valid');
    end
%     db(j) = sum(delta_3d(:)) / size(delta_3d, 3);
end
if strcmp(optimizer,'sgd')
%     conv2d_model.db = db;
    conv2d_model.dParams{1} = reshape(dk,[kernelsize*kernelsize,inmaps_num*outmaps_num]);
elseif strcmp(optimizer,'moment')
    alpha = ops.alpha;
    conv2d_model.dParams{1} = alpha .* conv2d_model.dParams{1} + ...
        reshape(dk,[kernelsize*kernelsize,inmaps_num*outmaps_num]);
%     conv2d_model.db = alpha .* conv2d_model.db + db;
end

%% output and record
conv2d_model.delta = delta_in;

end
function X = rot180(X)
    X = flipdim(flipdim(X, 1), 2);
end