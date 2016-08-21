function batchnorm_model = batchnorm_update(batchnorm_model,ops)
% forward of fullLinear layer
% model: 
% ops: option
% lichao , 20160717
%

%% params
% h = batchnorm_model.h;
% x = batchnorm_model.x;
% delta = batchnorm_model.delta;
learningrate = ops.learningrate;
gamma = batchnorm_model.gamma;
beta = batchnorm_model.beta;
dgamma = batchnorm_model.dgamma;
dbeta = batchnorm_model.dbeta;
%% update
gamma = gamma - learningrate * dgamma;
beta = beta - learningrate * dbeta;

%% output and record
batchnorm_model.gamma = gamma;
batchnorm_model.beta = beta;
end




