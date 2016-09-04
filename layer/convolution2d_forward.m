function conv2d_model = convolution2d_forward(conv2d_model, x)
% forward of convolution2d layer
% model: 
% x : input
% lichao , 20160717

%% params
inmaps_num = conv2d_model.inmaps_num;
outmaps_num = conv2d_model.outmaps_num;
inmap_size = conv2d_model.inmap_size;
outmap_size = conv2d_model.outmap_size;
kernelsize = conv2d_model.kernelsize;
is_same_size = conv2d_model.is_same_size;
order = conv2d_model.order;
k = reshape(conv2d_model.Params{1},[kernelsize, kernelsize,inmaps_num, outmaps_num]);

%assert (inmaps_num == size(x, 3), 'the first layer type must be i ');
%% process
%  !!below can probably be handled by insane matrix operations
num_sample = size(x,2);
% transfer x
x_2d = x;
switch (order)
    case 'whcn'
        size_x = [inmap_size, inmaps_num, num_sample];
        size_h = [outmap_size, outmaps_num, num_sample];
    case 'wchn'
        size_x = [inmap_size(1), inmaps_num, inmap_size(2), num_sample];
        size_h = [outmap_size(1), outmaps_num, outmap_size(2), num_sample];
    otherwise
        error('the M bigger than 10');
end
x = reshape(x_2d,size_x);
h = zeros(size_h);

if strcmp(is_same_size,'false')
    ops_conv = 'valid';
    batch_outonemap_size = [outmap_size, num_sample];
%     batch_inonemap_zerosend_size = size_x([1 2 4]);
else
    ops_conv = 'same';
    batch_outonemap_size = [inmap_size, num_sample];
%     batch_inonemap_zerosend_size = size_x([1 2 4]) + [kernelsize-1 kernelsize-1 0];
%     dim_start = (kernelsize-1)/2 + 1;
%     dim_end = batch_inonemap_zerosend_size(1) - (kernelsize-1)/2;
end
batch_inonemap_size = [inmap_size, num_sample];

for j = 1 :outmaps_num   %  for each output map
    %  create temp output map
    z = zeros(batch_outonemap_size);
    for i = 1 : inmaps_num   %  for each input map
%          if strcmp(is_same_size,'true') % 吧 x 四周变大
%             x_zerosend = single(fillin_value.*ones(batch_inonemap_zerosend_size));
%             x_zerosend(dim_start:dim_end, dim_start:dim_end, :) = x(:,:,i,:);
%          else
%             x_zerosend = reshape(x(:,:,i,:),batch_inonemap_zerosend_size);
%          end
        switch (order)
            case 'whcn'
                x_loc = x(:,:,i,:);
            case 'wchn'
                x_loc = x(:,i,:,:);
        end
        x_zerosend = reshape(x_loc, batch_inonemap_size);
        z = z + convn(x_zerosend, k(:,:,i,j), ops_conv);
    end
    %  add bias, pass through nonlinearity
    switch (order)
        case 'whcn'
%             h(:,:,j,:) = z + b(j);
            h(:,:,j,:) = z;
        case 'wchn'
%             h(:,j,:,:) = z + b(j);
            h(:,j,:,:) = z;
    end
end

size_h_out = [prod([outmap_size, outmaps_num]), num_sample];
h = reshape(h,size_h_out);

%% output and record
conv2d_model.x = x_2d;
conv2d_model.h = h;

end
