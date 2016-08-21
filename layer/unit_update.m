function unit = unit_update(unit, ops)
% update of unit layer
% used first in CNN
% lichao , 20160718

%% params
layer_type = unit.type;
% unit.layers{1}.a{outmaps_num} = x;

%% update 
switch layer_type
    case 'conv2d' % convolustion 2d
        unit = convolution2d_update(unit,ops);
        
    case 'pool2d'  % pooling 2d
        unit = pooling2d_update(unit,ops);
        
    case  'full' % FullLinear
        unit = fullLinear_update(unit,ops);
        
    case  'linear' % linear
        unit = linear_update(unit,ops);    
    
    case  'bias' % bias
        unit = bias_update(unit,ops);   
        
    case  'batchnorm' % bias
        unit = batchnorm_update(unit,ops);  
        
    otherwise
        error('the type of layer is UNDIFEND');
end


end
