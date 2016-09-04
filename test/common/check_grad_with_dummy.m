function check_grad_with_dummy(layer, x, y, p)
% used to check grad by stack a unit|stack layer with linear layer and a cost layer
% lichao, 20160830

%% (0) default paramters
if nargin < 4 % Linear
    le_tol = 1e-11;
    ge_tol = 1e-16;
    epsilon = 1e-7;
    iterations = 1;
    verbose = 'true';
    add_linear = 'false';
else
    le_tol = p.le_tol;
    ge_tol = p.ge_tol;
    epsilon = p.epsilon;
    iterations = p.iterations;
    verbose = p.verbose;
    add_linear = p.add_linear;
end

sampling_ratio = 1;

%% (1)
% nnet = cnn;
cuml_diff = 0;
indim = size(x, 1);
outdim = size(y, 1);
for i = 1:iterations
    dummy_nnet = build_dummy_nnet(layer, indim, outdim,add_linear);
    %sampling ratio set to 1.0 to process all params
        diff = check_grad(dummy_nnet, x, y, sampling_ratio, epsilon, verbose);
        cuml_diff = cuml_diff + diff;
end

%% Collect average statistic over all iterations.
avg_diff = cuml_diff/iterations;
disp([ 'Average Diff: ', num2str(avg_diff),' LE Tol: ', num2str( le_tol), ...
    ' GE Tol: ', num2str(ge_tol)]);

assert(avg_diff < le_tol, 'Too big');
assert(avg_diff > ge_tol, 'Too small');
    
end

function  nnet = build_dummy_nnet(layer, indim, outdim, add_linear)
    linear_conf.indim = indim;
    linear_conf.outdim = indim;
    linear_model = linear_set(linear_conf);
    linear_model = linear_initial(linear_model);
    %
    % MSE
    mse_cost_conf.indim = outdim;
    mse_cost_model = mse_cost_set(mse_cost_conf);
    mse_cost_model = mse_cost_initial(mse_cost_model);
    %
    % CE
%     ce_cost_conf.indim = outdim;
%     ce_cost_model = ce_cost_set(ce_cost_conf);
%     ce_cost_model = ce_cost_initial(ce_cost_model);
    %
    if isfield(layer,'is_cost') && strcmp(layer.is_cost,'true') 
        if strcmp(add_linear,'true')
            nnet.layers = {linear_model, layer};
            nnet.struct = ['linear ', layer.type];
        else
            nnet.layers = {layer};
            nnet.struct = layer.type;
        end
    else
        if strcmp(add_linear,'true')
            nnet.layers = {linear_model, layer, mse_cost_model};
            nnet.struct = ['linear ', layer.type, ' mse'];
        else
            nnet.layers = {layer, mse_cost_model};
            nnet.struct = [layer.type, ' mse'];
        end
    end
    nnet.layer_num = length(nnet.layers);
end
    