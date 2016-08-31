function diff = check_grad(nnet, x, y, sampling_ratio, epsilon)
% used to check the gradiant of nnet's parameters
% lichao,20160830
%

%% £¨0£© default paramters
if nargin < 5
     epsilon = single(1e-4);
end
if nargin < 4
     sampling_ratio = 1.0;
end
verbose = 'false'; 

%% (1) get original Params and dParams
[cost_ori, nnet] = cost_and_gradient(nnet, x, y);
nnet_ori = nnet;

%% (2) get the index of all Params and dParams
index_params = nnet_indexing_params(nnet);

%% (3) 
diff = single(0.0);
tot = single(0.0);
count_check = 0;
for sublayer_count = 1: length(index_params)
    [Params_loc, dParams_loc] = stack_extract_params(nnet, index_params{sublayer_count});
    for n = 1 : length(Params_loc)
        P_l = Params_loc{n};
        dP_l = dParams_loc{n};
        [I, J] = size(P_l);
        for i = 1: I
            for j = 1: J
                %Sample parameters according to the randomly generated number
                if (sampling_ratio ~= 1.0 && rand() > sampling_ratio)
                    continue;
                end
                count_check = count_check + 1;                
                orig_val = P_l(i,j);
                back_grad = dP_l(i,j);

                P_l(i,j) = orig_val + epsilon;
%                 nnet = nnet_ori;
                nnet = stack_stuff_param(nnet, index_params{sublayer_count},P_l, n);
                costP = cost_and_gradient(nnet, x, y);

                P_l(i,j) = orig_val - epsilon;
%                 nnet = nnet_ori;
                nnet = stack_stuff_param(nnet, index_params{sublayer_count},P_l, n);
                costN = cost_and_gradient(nnet, x, y);

                P_l(i,j) = orig_val;
                nnet = stack_stuff_param(nnet, index_params{sublayer_count},P_l, n);

                num_grad = (costP - costN) / (2. * epsilon);
                diff = diff + (num_grad - back_grad)^2;

                if strcmp(verbose,'true')
                    abs_diff = abs(num_grad - back_grad) / abs(back_grad);
                    if (abs_diff > 1e-5)
                        disp(['Param:', index_params{sublayer_count}, '::', ...
                            '(', num2str(i),',',num2str(j),') ', num2str(orig_val)]);
                        disp(['AbsDiff: ',num2str(abs_diff)]);
                        disp(['Grad: ',num2str(back_grad),'  NumGrad: ', num2str(num_grad)]);
                    end
                end
                tot = tot + back_grad^2;
            end
        end
    end
end

if (diff == 0 && tot == 0)
    diff = 0;
else
    diff = diff/tot;
end

if strcmp(verbose,'true')
	disp(['Diff:  ',num2str(diff)]);
	disp(['Number of parameters actually checked = ', num2str(count_check)]);
end
            
end





