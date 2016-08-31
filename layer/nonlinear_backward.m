function nonlinear_model = nonlinear_backward(nonlinear_model, delta)
% backward of pooling2d layer
% model: 
% x : input
% lichao , 20160717

%% params
nonlinearity = nonlinear_model.nonlinearity;
h = nonlinear_model.h;
%% process
%  feedbackward into output perceptrons
 switch nonlinearity
	case 'sigmoid'
		delta_in = delta .* h .* (1 - h);
        
	case 'linear'
		delta_in = delta;
        
	case 'softmax'
        D = h.*delta; % D = H .* E (in matlab notation)
        pe_vec = diag(h' * delta); %pe_vec.AddDiagMatMat(1.0, H, kNoTrans, E, kTrans, 0.0);
        delta_in = D - bsxfun(@times, h, pe_vec'); %D = D - diag(pe_vec) * H

     case 'tanh'
         delta_in = delta .* (1 - h.*h);
         
     case 'relu'
         h_post = zeros(size(h));
         h_post(find(h>0)) = 1;
         delta_in = delta .* h_post;
         
    otherwise
        error('the type of nonlinearity is UNDIFEND');
end

%% output and record
nonlinear_model.delta = delta_in;

end
