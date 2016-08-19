function bias_model = bias_initial(bias_model)
% used to initialize the bias model
% params [in]
%    struct('name', 'linear', 'class', 'unit', 'indim', 1024, 'outdim', 10 %linear layer
% parms [out]
%    add:  W
%lichao 20160717

%% used params to initialize
outdim = bias_model.outdim;

%%
b = single(zeros(outdim, 1));
db = single(zeros(outdim, 1));

%% (3) put into the struct
% 
bias_model.b = b;
bias_model.db = db;
bias_model.type = 'bias';
bias_model.class = 'unit';
bias_model.update = 'true';
bias_model.dim = 1;
end
