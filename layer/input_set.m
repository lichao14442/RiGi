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
input_model.inmaps_num = conf.inmaps_num;
input_model.inmap_size = conf.inmap_size;

%%
if isfield(conf,'name')
    input_model.name = conf.name;
else
    input_model.name = 'none';
end

