function input_model = input_initial(input_model)
% used to initialize the input model
% params [in]
%    struct('name', 'linear', 'class', 'unit', 'indim', 1024, 'outdim', 10 %linear layer
% parms [out]
%    add:  W
%lichao 20160717

%% used params to initialize
inmaps_num = input_model.inmaps_num;
inmap_size = input_model.inmap_size;
indim = input_model.indim;

%%
input_model.outmaps_num = inmaps_num;
input_model.outmap_size = inmap_size;
input_model.outdim = indim;

%% (3) put into the struct
% 
input_model.type = 'input';
input_model.class = 'unit';
input_model.update = 'false';
input_model.is_cost = 'false';

end
