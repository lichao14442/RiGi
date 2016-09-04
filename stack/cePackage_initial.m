function cePack_model = cePackage_initial(cePack_model)
% used to initialize the affine model
% params [in]
%    struct('name', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160904

%% used params to initialize
indim = cePack_model.indim;
outdim = cePack_model.outdim;
batch_normalized = cePack_model.batch_normalized;

%% initial for each layer
affine_conf.indim = indim;
affine_conf.outdim = outdim;
affine_conf.batch_normalized = batch_normalized;
affine_conf.name = [cePack_model.name,'->affine'];
affine_model = affine_set(affine_conf);
affine_model = affine_initial(affine_model);
%
ce_conf.indim = outdim;
ce_conf.name = [cePack_model.name,'->ce'];
ce_model = ce_cost_set(ce_conf);
ce_model = ce_cost_initial(ce_model);
%
% set_stack
layers = {affine_model, ce_model};

%% (3) put into the struct
cePack_model = stack_build(layers, cePack_model);
cePack_model.type = 'cePack';

end
