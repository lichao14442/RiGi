创建一个stack网络层：
	（1）set ： 需要变的变量在这里赋值
	（2）initial： 不需要变的变量在这里赋值，同时初始化矩阵
	（3）nnet_setup.m， 修改它，使其认识新层的type
	
Creat a stack layer: need to write(or modify) 3 files:
		(1) xxx_set: set the variable changing in different sampes
		(2) xxx_initial: set the variable unchanging in different samples, meanwhile, the initial the layer params.
		(3) nnet_setup: modify it in order to known the new layer.
