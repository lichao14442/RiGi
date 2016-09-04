function pool2d_model = pooling2d_set(conf,pool2d_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    pool2d_model = [];
end

pool2d_model.inmaps_num = conf.inmaps_num;
pool2d_model.outmaps_num = conf.inmaps_num;
pool2d_model.inmap_size = conf.inmap_size;
pool2d_model.scale = conf.scale;

%%
if isfield(conf,'name')
    pool2d_model.name = conf.name;
else
    pool2d_model.name = 'pooling2d';
end

if isfield(conf, 'method')
     pool2d_model.method = conf.method;
else
    pool2d_model.method = 'max';
end

if isfield(conf,'order')
    pool2d_model.order = conf.order;
else
    pool2d_model.order = 'whcn';
end

