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
        D = h.*delta;
        pe_vec = diag(h' * delta);
        delta_in = D - bsxfun(@times, h, pe_vec'); 
%         delta_in = D - h * diag(pe_vec);
          
% kaldi:  const CuMatrixBase<Real> &P(value), &E(diff);
%         CuMatrixBase<Real> &D(*this);
%         D.CopyFromMat(P);
%         D.MulElements(E);
%         // At this point, D = P .* E (in matlab notation)
%         CuVector<Real> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
%         pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);
% 
%         D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.  
        
    
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
