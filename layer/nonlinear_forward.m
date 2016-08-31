function nonlinear_model = nonlinear_forward(nonlinear_model, x)
% forward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
nonlinearity = nonlinear_model.nonlinearity;

%% process
%  feedforward into output perceptrons
 switch nonlinearity
	case 'sigmoid'
		h = sigm(x);
        
	case 'linear'
		h = x;
        
	case 'softmax'
        h = exp(bsxfun(@minus, x, max(x,[],1)));
        h = bsxfun(@rdivide, h, sum(h, 1)); 
        
     case 'tanh'
         h = tanh(x);
         
     case 'relu'
         h = max(0, x);
         
    otherwise
        error('the type of nonlinearity is UNknown');
        
end

%% output and record
nonlinear_model.x = x;
nonlinear_model.h = h;

end
