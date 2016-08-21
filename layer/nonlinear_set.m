function nonlinear_model = nonlinear_set(conf,nonlinear_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    nonlinear_model = [];
end
nonlinear_model.outdim = conf.indim;
nonlinear_model.outdim = conf.indim;

%%
if isfield(conf,'nonlinearity')
    nonlinear_model.nonlinearity = conf.nonlinearity;
else
    nonlinear_model.nonlinearity = 'sigmoid';
end

if isfield(conf,'name')
    nonlinear_model.name = conf.name;
else
    nonlinear_model.name = 'none';
end

if isfield(conf,'need_convert_dim')
    nonlinear_model.need_convert_dim = conf.need_convert_dim;
else
    nonlinear_model.need_convert_dim = 'false';
end


