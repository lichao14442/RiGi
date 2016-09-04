function pool2d_model = pooling2d_backward(pool2d_model, ops, delta)
% forward of pooling2d layer
% model: 
% ops : option
% lichao , 20160717

%% params
inmaps_num = pool2d_model.inmaps_num;
inmap_size = pool2d_model.inmap_size;
outmaps_num = pool2d_model.outmaps_num;
outmap_size = pool2d_model.outmap_size;
order = pool2d_model.order;
scale = pool2d_model.scale;
method = pool2d_model.method;
% h = pool2d_model.h;
% x = pool2d_model.x;
% delta = pool2d_model.delta;
% optimizer = ops.optimizer;
% b = pool2d_model.b;
% db = pool2d_model.db;
% delta_in = cell(1,outmaps_num);
%assert (inmaps_num == size(x, 3), 'the first layer type must be i ');
%% process
%  !!below can probably be handled by insane matrix operations
num_sample = size(delta,2);

if num_sample == 1
    ddim = [scale scale];
else
    ddim = [scale scale 1];
end

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

delta = reshape(delta, size_h);
delta_in = zeros(size_x);

batch_inonemap_size = [inmap_size, num_sample];
batch_outonemap_size = [outmap_size, num_sample];
%
for j = 1 :inmaps_num   %  for each output map
    switch (order)
        case 'whcn'
            delta_loc = delta(:,:,j,:);
        case 'wchn'
            delta_loc = delta(:,j,:,:);
    end
    delta_3d = reshape(delta_loc, batch_outonemap_size);
    if strcmp(method,'average')
        switch (order)
            case 'whcn'
                delta_in(:,:,j,:) = expand(delta_3d, ddim) / scale^2;
            case 'wchn'
                delta_in(:,j,:,:) = expand(delta_3d, ddim) / scale^2;
        end
    elseif strcmp(method,'max')
        switch (order)
            case 'whcn'
                maps_idxmax = reshape(pool2d_model.allmaps_idxmax(:,:,j,:),batch_inonemap_size);
                delta_in(:,:,j,:) = expand(delta_3d, ddim) .* maps_idxmax;
            case 'wchn'
                maps_idxmax = reshape(pool2d_model.allmaps_idxmax(:,j,:,:),batch_inonemap_size);
                delta_in(:,j,:,:) = expand(delta_3d, ddim) .* maps_idxmax;
        end
    else
       error('the other mothed is not yet supported'); 
    end
end

size_x_out = [prod([inmap_size, inmaps_num]), num_sample];
delta_in = reshape(delta_in,size_x_out);

%% compute giadient

%% output and record
pool2d_model.delta = delta_in;

end
