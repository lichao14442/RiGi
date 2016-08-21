function conv2d_model = convolution2d_load(model_dir,start_num,conv2d_model)
% used to load conv2d parames
% lichao 20160729
%
if nargin == 0
    model_dir = 'testdata/npy_model';
    start_num = 0;
end

if (strfind(model_dir,'txt'))
    error('not support ')
    
elseif (strfind(model_dir,'npy'))
    Wfile  = fullfile(model_dir, ['params', num2str(start_num),'.npy']);
    W = single(readNPY(Wfile));   
end

outmaps_num_data = size(W,2);

%£¨2£©init a conv2d_model,
inmaps_num = conv2d_model.inmaps_num;
outmaps_num = conv2d_model.outmaps_num;
kernelsize = conv2d_model.kernelsize;
assert(outmaps_num_data == outmaps_num, 'error in outmaps_num');
k = reshape(W,[kernelsize,kernelsize,inmaps_num,outmaps_num]);

%(3) put into the struct
conv2d_model.k = k;








