function unit = unit_update(unit, ops)
% update of unit layer
% used first in CNN
% lichao , 20160718

%% params
learningrate = ops.learningrate;

%%
num_params = length(unit.Params);
for i = 1: num_params
    unit.Params{i} = unit.Params{i} - learningrate * unit.dParams{i}; 
end

end
