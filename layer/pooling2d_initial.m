function pool2d_model = pooling2d_initial(pool2d_model)
% used to initialize the convolution2D model
% params [in]
%    struct('type', 's','inmaps_num', 6, 'inmap_size', [28 28], 
%            'scale', 2) %sub sampling layer
% parms [out]
%    add:  outmap_size, k, b
%lichao 20160717
%这个函数中 scale 完全可以写成2d的！todo

%% used params to initialize
inmaps_num = pool2d_model.inmaps_num;
inmap_size = pool2d_model.inmap_size;
scale = pool2d_model.scale;

%%
outmap_size = inmap_size ./ scale;
assert(all(floor(outmap_size)==outmap_size), 'Layer pooling2d size must be integer. Actual: ');

outdim = prod(outmap_size)*inmaps_num;
%% (3) put into the struct
pool2d_model.outdim = outdim;
pool2d_model.outmap_size = outmap_size;
% pool2d_model.b = b;
% pool2d_model.db = db;
pool2d_model.outmaps_num = inmaps_num;
pool2d_model.type = 'pool2d';
pool2d_model.class = 'unit';
pool2d_model.update = 'false';
pool2d_model.is_cost = 'false';
end
