function full_model = fullLinear_initial(full_model)
% used to initialize the fulllinear model
% params [in]
%    struct('type', 'f','indim', 1024, 'outdim', 10 %full conation linear layer
% parms [out]
%    add:  W, b
%lichao 20160717
%这个函数中 kernelsize 完全可以写成2d的！todo

%% used params to initialize
indim = full_model.indim;
outdim = full_model.outdim;

%%
b = single(zeros(outdim, 1));
W = single(rand(outdim, indim) - 0.5) * 2 * sqrt(6 / (outdim + indim));
dW = single(zeros(outdim, indim));
db = single(zeros(outdim, 1));

%% (3) put into the struct
if ~isfield(full_model, 'need_convert_dim')
    full_model.need_convert_dim = 'false';
end
%
if ~isfield(full_model, 'activation_function')
    full_model.activation_function = 'sigm';
end

%
if ~isfield(full_model, 'dropoutFraction')
    full_model.dropoutFraction = 0;
end


full_model.W = W;
full_model.b = b;
full_model.dW = dW;
full_model.db = db;
full_model.type = 'full';
full_model.class = 'unit';
full_model.update = 'true';
full_model.dim = 1;
end
