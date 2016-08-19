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
if isfield(conf,'name')
    linear_model.name = conf.name;
else
    linear_model.name = 'none';
end
if isfield(conf,'need_convert_dim')
    linear_model.need_convert_dim = conf.need_convert_dim;
else
    linear_model.need_convert_dim = 'false';
end


