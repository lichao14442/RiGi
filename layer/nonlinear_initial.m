function nonlinear_model = nonlinear_initial(nonlinear_model)
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
nonlinear_model.type = 'nonlinear';
nonlinear_model.class = 'unit';
nonlinear_model.update = 'false';
nonlinear_model.is_cost = 'false';
nonlinear_model.dim = 1;
end
