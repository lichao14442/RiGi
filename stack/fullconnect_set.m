function full_model = fullconnect_set(conf,full_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    full_model = [];
end

full_model.indim = conf.indim;
full_model.outdim = conf.outdim;

%%
if isfield(conf,'nonlinearity')
    full_model.nonlinearity = conf.nonlinearity;
else
    full_model.nonlinearity = 'sigm';
end

if isfield(conf,'name')
    full_model.name = conf.name;
else
    full_model.name = 'none';
end

if isfield(conf,'need_convert_dim')
    full_model.need_convert_dim = conf.need_convert_dim;
else
    full_model.need_convert_dim = 'false';
end

if isfield(conf,'batch_normalized')
    full_model.batch_normalized = conf.batch_normalized;
else
    full_model.batch_normalized = 'false';
end

