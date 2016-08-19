function mse_model = mse_cost_initial(mse_model)
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
mse_model.type = 'mse-cost';
mse_model.class = 'unit';
mse_model.update = 'false';
mse_model.dim = 1;
end
