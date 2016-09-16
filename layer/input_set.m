function input_model = input_set(conf,input_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    input_model = [];
end
input_model.indim = conf.indim;
input_model.outdim = conf.indim;

%%
if isfield(conf,'inmap_size')
    input_model.inmap_size = conf.inmap_size;
else
    input_model.inmap_size = [1,1];
end

if isfield(conf,'order')
    input_model.order = conf.order;
else
    input_model.order = 'whcn';
end

if isfield(conf,'inmaps_num')
    input_model.inmaps_num = conf.inmaps_num;
else
    input_model.inmaps_num = conf.indim;
end

if isfield(conf,'name')
    input_model.name = conf.name;
else
    input_model.name = 'input';
end

