function bias_model = bias_initial(bias_model)
% used to initialize the bias model
% params [in]
%    struct('name', 'linear', 'class', 'unit', 'indim', 1024, 'outdim', 10 %linear layer
% parms [out]
%    add:  W
%lichao 20160717

%% used params to initialize
indim = bias_model.indim;
axis_to_norm = bias_model.axis_to_norm;
inmap_size = bias_model.inmap_size;
inmaps_num = bias_model.inmaps_num;

%%
if axis_to_norm == 0
    param_dim = indim;
else
    param_dim = inmaps_num;
end
%%
% b = single(zeros(param_dim, 1));
% db = single(zeros(param_dim, 1));
b = zeros(param_dim, 1);
db = zeros(param_dim, 1);
%% (3) put into the struct
% 
bias_model.outdim = indim;
bias_model.outmap_size = inmap_size;
bias_model.outmaps_num = inmaps_num;
bias_model.param_dim = param_dim;
bias_model.Params = {b};
bias_model.dParams = {db};
bias_model.type = 'bias';
bias_model.class = 'unit';
bias_model.update = 'true';
bias_model.is_cost = 'false';
bias_model.dim = 1;
end
