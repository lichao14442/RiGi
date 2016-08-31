function linear_model = linear_initial(linear_model)
% used to initialize the fulllinear model
% params [in]
%    struct('name', 'linear', 'indim', 1024, 'outdim', 10 %linear layer
% parms [out]
%    add:  W
%lichao 20160717

%% used params to initialize
indim = linear_model.indim;
outdim = linear_model.outdim;

%%
% W = single(rand(outdim, indim) - 0.5) * 2 * sqrt(6 / (outdim + indim));
% dW = single(zeros(outdim, indim));
W = (rand(outdim, indim) - 0.5) * 2 * sqrt(6 / (outdim + indim));
dW = zeros(outdim, indim);

%% (3) put into the struct
% 
linear_model.Params = {W};
linear_model.dParams = {dW};
linear_model.type = 'linear';
linear_model.class = 'unit';
linear_model.update = 'true';
linear_model.is_cost = 'false';
linear_model.dim = 1;
end
