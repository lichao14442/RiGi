function cePack_model = cePackage_set(conf,cePack_model)
% used to set paramters of linear model
% input :
%       conf: the struct configure
%       lstmp: the model
%
% lichao 20160904


%conf = struct('type', 'l', 'need_act_h', 1,'clip_gradient',clip_gradient);
if nargin < 2
    cePack_model = [];
end

cePack_model.indim = conf.indim;
cePack_model.outdim = conf.outdim;

%%

if isfield(conf,'name')
    cePack_model.name = conf.name;
else
    cePack_model.name = 'cePack';
end

if isfield(conf,'batch_normalized')
    cePack_model.batch_normalized = conf.batch_normalized;
else
    cePack_model.batch_normalized = 'false';
end

