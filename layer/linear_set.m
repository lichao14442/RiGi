function linear_model = linear_set(conf,linear_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    linear_model = [];
end

linear_model.indim = conf.indim;
linear_model.outdim = conf.outdim;

%%
if isfield(conf,'inmap_size')
    linear_model.inmap_size = conf.inmap_size;
else
    linear_model.inmap_size = [1,1];
end

if isfield(conf,'inmaps_num')
    linear_model.inmaps_num = conf.inmaps_num;
else
    linear_model.inmaps_num = conf.indim;
end

if isfield(conf,'name')
    linear_model.name = conf.name;
else
    linear_model.name = 'linear';
end


