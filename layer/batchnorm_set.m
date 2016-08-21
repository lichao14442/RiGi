function batchnorm_model = batchnorm_set(conf,batchnorm_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160725


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    batchnorm_model = [];
end

batchnorm_model.indim = conf.indim;
batchnorm_model.outdim = conf.indim;

% batchnorm_model.layer_index = conf.layer_index;
% batchnorm_model.model_dir = conf.model_dir;

%%

if isfield(conf,'name')
    batchnorm_model.name = conf.name;
else
    batchnorm_model.name = 'batchnorm';
end

if isfield(conf,'bn_eval_stats')
    batchnorm_model.bn_eval_stats = conf.bn_eval_stats;
else
    batchnorm_model.bn_eval_stats = 100;
end

if isfield(conf,'axis_to_norm')
    batchnorm_model.axis_to_norm = conf.axis_to_norm;
else
    batchnorm_model.axis_to_norm = 0;
end

if isfield(conf,'inmap_size')
    batchnorm_model.inmap_size = conf.inmap_size;
else
    batchnorm_model.inmap_size = [conf.indim, 1];
end

if isfield(conf,'inmaps_num')
    batchnorm_model.inmaps_num = conf.inmaps_num;
else
    batchnorm_model.inmaps_num = 1;
end

if isfield(conf,'test_mode')
    batchnorm_model.test_mode = conf.test_mode;
else
    batchnorm_model.test_mode = 'false';
end
