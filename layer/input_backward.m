function input_model = input_backward(input_model, ops, delta)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160831


inmaps_num = input_model.inmaps_num;
inmap_size = input_model.inmap_size;
indim = input_model.indim;
order = input_model.order;

%% output and record
input_model.delta = delta;

[num_sample] = size(delta,2);
% transfer x
delta_2d = delta;
ordred_size = [inmap_size, inmaps_num, num_sample];
switch (order)
    case 'whcn'
        ops_order = [1,2,3,4];
    case 'wchn'
        ops_order = [1,3,2,4];
    case 'cwhn'
        ops_order = [3,1,2,4];
    otherwise
        error('the original order is NOT support!');
end
% order: WHCN
% ordred_size = [inmap_size, inmaps_num, num_sample];

delta = reshape(delta_2d,ordred_size);
delta_ordered = permute(delta, ops_order);

delta_in = reshape(delta_ordered,indim,num_sample);
input_model.delta = delta_in;
end
