function conv2d_model = convolution2d_initial(conv2d_model)
% used to initialize the convolution2D model
% params [in]
%    struct('type', 'c','inmaps_num', 1, 'inmap_size', [28 28], 
%           'outmaps_num', 6, 'kernelsize', 5) %convolution layer
% parms [out]
%    add:  outmap_size, k, b
%lichao 20160717
%
%这个函数中 kernelsize 完全可以写成2d的！todo

%% used params to initialize
inmaps_num = conv2d_model.inmaps_num;
outmaps_num = conv2d_model.outmaps_num;
inmap_size = conv2d_model.inmap_size;
kernelsize = conv2d_model.kernelsize;
is_same_size = conv2d_model.is_same_size;
% k = cell(inmaps_num,1);
% dk = cell(inmaps_num,1);
%%
if strcmp(is_same_size,'false')
    outmap_size = inmap_size - kernelsize + 1;
else
    outmap_size = inmap_size;
end
fan_in = inmaps_num * kernelsize ^ 2;
fan_out = outmaps_num * kernelsize ^ 2;
scale =  2 * sqrt(6 / (fan_in + fan_out));
k = single(rand(kernelsize,kernelsize,inmaps_num, outmaps_num) - 0.5) .* scale; %rand 是 0.5均值的。 randn是 0均值的。
dk = single(zeros(kernelsize,kernelsize,inmaps_num, outmaps_num));
% b = single(zeros(1,outmaps_num));
% db = single(zeros(1,outmaps_num));

outdim = prod(outmap_size)*outmaps_num;
%% (3) put into the struct
conv2d_model.outdim = outdim;
conv2d_model.outmap_size = outmap_size;
conv2d_model.k = k;
% conv2d_model.b = b;
conv2d_model.dk = dk;
% conv2d_model.db = db;
conv2d_model.type = 'conv2d';
conv2d_model.class = 'unit';
conv2d_model.update = 'true';
conv2d_model.is_cost = 'false';
end
