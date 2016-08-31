function pool2dpack_model = pool2dPackage_set(conf,pool2dpack_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160728


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    pool2dpack_model = [];
end

pool2dpack_model.inmaps_num = conf.inmaps_num;
pool2dpack_model.outmaps_num = conf.inmaps_num;
pool2dpack_model.inmap_size = conf.inmap_size;
pool2dpack_model.scale = conf.scale;

%%

if isfield(conf,'name')
    pool2dpack_model.name = conf.name;
else
    pool2dpack_model.name = 'none';
end

if isfield(conf,'batch_normalized')
    pool2dpack_model.batch_normalized = conf.batch_normalized;
else
    pool2dpack_model.batch_normalized = 'false';
end

if isfield(conf,'order')
    pool2dpack_model.order = conf.order;
else
    pool2dpack_model.order = 'whcn';
end
