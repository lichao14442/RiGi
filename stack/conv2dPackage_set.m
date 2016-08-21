function conv2dpack_model = conv2dPackage_set(conf,conv2dpack_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    conv2dpack_model = [];
end

conv2dpack_model.inmaps_num = conf.inmaps_num;
conv2dpack_model.outmaps_num = conf.outmaps_num;
conv2dpack_model.inmap_size = conf.inmap_size;
conv2dpack_model.kernelsize = conf.kernelsize;

%%
if isfield(conf,'nonlinearity')
    conv2dpack_model.nonlinearity = conf.nonlinearity;
else
    conv2dpack_model.nonlinearity = 'relu';
end

if isfield(conf,'name')
    conv2dpack_model.name = conf.name;
else
    conv2dpack_model.name = 'none';
end

if isfield(conf,'batch_normalized')
    conv2dpack_model.batch_normalized = conf.batch_normalized;
else
    conv2dpack_model.batch_normalized = 'false';
end

if isfield(conf,'is_same_size')
    conv2dpack_model.is_same_size = conf.is_same_size;
else
    conv2dpack_model.is_same_size = 'true';
end

if isfield(conf,'order')
    conv2dpack_model.order = conf.order;
else
    conv2dpack_model.order = 'whcn';
end

