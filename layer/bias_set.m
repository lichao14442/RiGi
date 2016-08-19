function bias_model = bias_set(conf,bias_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    bias_model = [];
end
bias_model.indim = conf.indim;
bias_model.outdim = conf.indim;

%%
if isfield(conf,'name')
    bias_model.name = conf.name;
else
    bias_model.name = 'none';
end

if isfield(conf,'need_convert_dim')
    bias_model.need_convert_dim = conf.need_convert_dim;
else
    bias_model.need_convert_dim = 'false';
end
