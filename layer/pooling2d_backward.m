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
scale = pool2d_model.scale;
method = pool2d_model.method;

%% process
num_sample = size(delta,2);

if num_sample == 1
    ddim = [scale scale];
else
    ddim = [scale scale 1];
end
size_x = [inmap_size, inmaps_num, num_sample];
size_h = [outmap_size, outmaps_num, num_sample];

delta = reshape(delta, size_h);
delta_in = zeros(size_x);

batch_inonemap_size = [inmap_size, num_sample];
batch_outonemap_size = [outmap_size, num_sample];
%
for j = 1 :inmaps_num   %  for each output map
    delta_loc = delta(:,:,j,:);
    delta_3d = reshape(delta_loc, batch_outonemap_size);
    if strcmp(method,'average')
        delta_in(:,:,j,:) = expand(delta_3d, ddim) / scale^2;
    elseif strcmp(method,'max')
        maps_idxmax = reshape(pool2d_model.allmaps_idxmax(:,:,j,:),batch_inonemap_size);
        delta_in(:,:,j,:) = expand(delta_3d, ddim) .* maps_idxmax;
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
