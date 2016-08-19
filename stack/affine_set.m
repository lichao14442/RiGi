function affine_model = affine_set(conf,affine_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160718


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    affine_model = [];
end

affine_model.indim = conf.indim;
affine_model.outdim = conf.outdim;

%%

if isfield(conf,'name')
    affine_model.name = conf.name;
else
    affine_model.name = 'none';
end


if isfield(conf,'need_convert_dim')
    affine_model.need_convert_dim = conf.need_convert_dim;
else
    affine_model.need_convert_dim = 'false';
end

if isfield(conf,'batch_normalized')
    affine_model.batch_normalized = conf.batch_normalized;
else
    affine_model.batch_normalized = 'false';
end

