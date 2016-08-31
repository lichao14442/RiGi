function ce_model = ce_cost_initial(ce_model)
% used to initialize the fulllinear model
% params [in]
%    struct('name', 'linear', 'indim', 1024, 'outdim', 10 %linear layer
% parms [out]
%    add:  W
%lichao 20160717

%% used params to initialize

%%

%% (3) put into the struct
% 
ce_model.type = 'ce-cost';
ce_model.class = 'unit';
ce_model.update = 'false';
ce_model.is_cost = 'true';
ce_model.dim = 1;
end
