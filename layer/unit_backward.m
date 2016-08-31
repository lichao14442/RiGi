function unit = unit_backward(unit, ops, delta)
% backward of unit layer
% used first in CNN
% lichao , 20160718

%% params
layer_type = unit.type;
% unit.layers{1}.a{outmaps_num} = x;

%% backward 
switch layer_type
    case 'input'
        unit = input_backward(unit, ops, delta);
        
    case 'conv2d' % convolustion 2d
        unit = convolution2d_backward(unit, ops, delta);
        
    case 'pool2d'  % pooling 2d
        unit = pooling2d_backward(unit, ops, delta);
        
    case  'full' % FullLinear
        unit = fullLinear_backward(unit, ops, delta);
        
    case  'linear' % linear
        unit = linear_backward(unit, ops, delta);    
    
    case  'bias' % bias
        unit = bias_backward(unit, ops, delta);   
            
    case  'batchnorm' % batchnorm
        unit = batchnorm_backward(unit, ops, delta);  
        
    case  'nonlinear' % nonlinear
        unit = nonlinear_backward(unit, delta);   
        
    case  'ce-cost' % ce-cost
        unit = ce_cost_backward(unit, delta);   
                
    case  'mse-cost' % mse-cost
        unit = mse_cost_backward(unit, delta);   
        
    otherwise
        error('the type of layer is UNDIFEND');
end


end
