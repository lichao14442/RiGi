创建一个网络层：
	（1）set ： 需要变的变量在这里赋值
	（2）initial： 不需要变的变量在这里赋值，同时初始化矩阵
	（3）forward： 前向传播 -> 写到unit_forward里面
	（4）backward：反向传播 -> 写到unit_fbackward里面
	（6）load: load参数，如果有有必要，主要用来debug和断点续训
	（7）nnet_setup.m， 修改它，使其认识新层的type
	
Creat a basic layer: need to write 5+ files:
		(1) xxx_set: set the variable changing in different sampes
		(2) xxx_initial: set the variable unchanging in different samples, meanwhile, the initial the layer params.
		(3) xxx_forward: forward. And write into unit_forward.m, meanwhile.
		(4) xxx_backward: backward. And write into unit_backward.m, meanwhile.
		(+6) xxx_load: load params, used to debug, and retrain from a cutpoit.
		(7) nnet_setup: modify it in order to known the new layer.
	
	
