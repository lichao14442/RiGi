function mse_model = mse_cost_set(conf,mse_model)
% used to set paramters of mse layer
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    mse_model = [];
end

mse_model.indim = conf.indim;
mse_model.outdim = conf.indim;
   
%%
if isfield(conf,'name')
    mse_model.name = conf.name;
else
    mse_model.name = 'none';
end
if isfield(conf,'need_convert_dim')
    mse_model.need_convert_dim = conf.need_convert_dim;
else
    mse_model.need_convert_dim = 'false';
end

