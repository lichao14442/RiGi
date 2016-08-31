function full_model = fullLinear_forward(full_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
W = full_model.W;
b = full_model.b;
activation_function = full_model.activation_function;
%assert (inmaps_num == size(x, 3), 'the first layer type must be i ');

%% process
%  feedforward into output perceptrons
 switch activation_function
	case 'sigm'
		h = sigm(W * x + repmat(b, 1, size(x, 2)));
	case 'linear'
		h = W * x + repmat(b, 1, size(x, 2));
	case 'softmax'
		h = W * x + repmat(b, 1, size(x, 2));
		h = exp(bsxfun(@minus, h, max(h,[],2)));
		h = bsxfun(@rdivide, h, sum(h, 2)); 
end

%% output and record
full_model.x = x;
full_model.h = h;

end
