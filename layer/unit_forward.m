function unit = unit_forward(unit, x)
% forward of unit layer
% used first in CNN
% lichao , 20160718

%% params
layer_type = unit.type;
% unit.layers{1}.a{outmaps_num} = x;

%% forward 
switch layer_type
    case 'input' % convolustion 2d
        unit = input_forward(unit, x);
        
    case 'conv2d' % convolustion 2d
        unit = convolution2d_forward(unit, x);
        
    case 'pool2d'  % pooling 2d
        unit = pooling2d_forward(unit, x);
        
    case  'full' % FullLinear
        unit = fullLinear_forward(unit, x);
        
    case  'linear' % linear
        unit = linear_forward(unit, x);    
    
    case  'bias' % bias
        unit = bias_forward(unit, x);   
        
    case  'batchnorm' % batchnorm
        unit = batchnorm_forward(unit, x);   
        
    case  'nonlinear' % nonlinear
        unit = nonlinear_forward(unit, x);   
        
    case  'ce-cost' % ce-cost
        unit = ce_cost_forward(unit, x);   
        
    case  'mse-cost' % mse-cost
        unit = mse_cost_forward(unit, x);   
        
    otherwise
        error('the type of layer is UNDIFEND');
end


end
