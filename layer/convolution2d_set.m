function conv2d_model = convolution2d_set(conf,conv2d_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    conv2d_model = [];
end

conv2d_model.inmaps_num = conf.inmaps_num;
conv2d_model.outmaps_num = conf.outmaps_num;
conv2d_model.inmap_size = conf.inmap_size;
conv2d_model.kernelsize = conf.kernelsize;

%%   
    
if isfield(conf,'name')
    conv2d_model.name = conf.name;
end

if isfield(conf,'need_convert_dim')
    conv2d_model.need_convert_dim = conf.need_convert_dim;
else
    conv2d_model.need_convert_dim = 'false';
end

if isfield(conf,'is_same_size')
    conv2d_model.is_same_size = conf.is_same_size;
else
    conv2d_model.is_same_size = 'true';
end

if isfield(conf,'order')
    conv2d_model.order = conf.order;
else
    conv2d_model.order = 'whcn';
end


