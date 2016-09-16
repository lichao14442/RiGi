function input_model = input_forward(input_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717


inmaps_num = input_model.inmaps_num;
inmap_size = input_model.inmap_size;
indim = input_model.indim;
order = input_model.order;
%% output and record
input_model.x = x;

%% reorder x 
[num_sample] = size(x,2);
% transfer x
x_2d = x;
switch (order)
    case 'whcn'
        original_size = [inmap_size, inmaps_num, num_sample];
        ops_order = [1,2,3,4];
    case 'wchn'
        original_size = [inmap_size(1), inmaps_num, inmap_size(2), num_sample];
        ops_order = [1,3,2,4];
    case 'cwhn'
        original_size = [inmaps_num, inmap_size(1), inmap_size(2), num_sample];   
        ops_order = [2,3,1,4];
    otherwise
        error('the original order is NOT support!');
end
% order: WHCN
% ordred_size = [inmap_size, inmaps_num, num_sample];

x = reshape(x_2d,original_size);
x_ordered = permute(x, ops_order);

h = reshape(x_ordered,indim,num_sample);
input_model.h = h;

end
