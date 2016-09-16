function pool2d_model = pooling2d_forward(pool2d_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717
% only support 4d tensor 'whcn' order, 20160915

%% params
inmaps_num = pool2d_model.inmaps_num;
inmap_size = pool2d_model.inmap_size;
outmaps_num = pool2d_model.outmaps_num;
outmap_size = pool2d_model.outmap_size;
scale = pool2d_model.scale;
method = pool2d_model.method;

%% process
num_sample = size(x,2);

x_2d = x;
size_x = [inmap_size, inmaps_num, num_sample];
size_h = [outmap_size, outmaps_num, num_sample];
x = reshape(x_2d,size_x);
h = zeros(size_h);
batch_inonemap_size = [inmap_size, num_sample];

for j = 1 :outmaps_num   %  for each output map
    x_loc = x(:,:,j,:);
    x_3d = reshape(x_loc, batch_inonemap_size);
    if strcmp(method,'average')
        z = convn(x_3d, ones(scale) / scale^2, 'valid');   %  !! replace with variable
        h(:,:,j,:) = z(1 : scale : end, 1 : scale : end, :);
    elseif strcmp(method,'max')
        [z, maps_idxmax] = max_inmaps(x_3d, scale);   %  !! replace with variable
        pool2d_model.allmaps_idxmax(:,:,j,:) = maps_idxmax;
        h(:,:,j,:) = z;
    else
       error('the other mothed is not yet supported'); 
    end
end

size_h_out = [prod([outmap_size, outmaps_num]), num_sample];
h = reshape(h,size_h_out);

%% output and record
pool2d_model.x = x;
pool2d_model.h = h;

end

function [z, maps_idxmax] = max_inmaps(x_3d, scale)
    [h, w, num_sample] = size(x_3d);
    maps_idxmax = zeros([h, w, num_sample]);
    filter_size = [scale, scale];
    h_out = ceil((h - filter_size(1)+1)/scale);
    w_out = ceil((w - filter_size(2)+1)/scale);
    batch_outonemap_size = [h_out, w_out, num_sample];
    z = zeros(batch_outonemap_size);
    %
    size_loc = [prod(filter_size), num_sample];
    for idx_w = 1: size(z,2)
       for idx_h = 1: size(z,1) 
           range_h = (idx_h-1)*scale+1:idx_h*scale;
           range_w = (idx_w-1)*scale+1:idx_w*scale;
           x_loc = x_3d(range_h, range_w, :);
           x_loc2 = reshape(x_loc,size_loc);
           [z(idx_h,idx_w,:),idx] = max(x_loc2);

           temp= zeros(size_loc);
           for idx_batch = 1: num_sample
               temp(idx(idx_batch),idx_batch) = 1;
           end

           maps_idxmax(range_h, range_w, :) = reshape(temp,[filter_size,num_sample] );
       end
    end
    
  
%     for idx_batch = 1: size(z,3)
%         for idx_w = 1: size(z,2)
%            for idx_h = 1: size(z,1) 
%                range_h = (idx_h-1)*scale+1:idx_h*scale;
%                range_w = (idx_w-1)*scale+1:idx_w*scale;
%                x_loc = x_3d(range_h, range_w, idx_batch);
%                [z(idx_h,idx_w,idx_batch),idx] = max(x_loc(:));
%                temp = single(zeros(filter_size));
%                temp(idx) = 1;
%                maps_idxmax(range_h, range_w, idx_batch) = temp;
%            end
%         end
%     end
end



