function conv2d_model = convolution2d_update(conv2d_model, ops)
% update of convolution2d layer
% model: 
% ops : option
% lichao , 20160717

%% params
% h = conv2d_model.h;
% x = conv2d_model.x;
% delta = conv2d_model.delta;
learningrate = ops.learningrate;
k = conv2d_model.k;
% b = conv2d_model.b;
dk = conv2d_model.dk;
% db = conv2d_model.db;
%assert (inmaps_num == size(x, 3), 'the first layer type must be i ');

%% backward delta
k = k - learningrate * dk;
% b = b - learningrate * db;
%% output and record
conv2d_model.k = k;
% conv2d_model.b = b;

end