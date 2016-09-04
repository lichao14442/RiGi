function ce_model = ce_cost_set(conf,ce_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    ce_model = [];
end

ce_model.indim = conf.indim;
ce_model.outdim = conf.indim;


%%
if isfield(conf,'name')
    ce_model.name = conf.name;
else
    ce_model.name = 'ce';
end

% if isfield(conf,'need_convert_dim')
%     ce_model.need_convert_dim = conf.need_convert_dim;
% else
%     ce_model.need_convert_dim = 'false';
% end

