����һ������㣺
	��1��set �� ��Ҫ��ı��������︳ֵ
	��2��initial�� ����Ҫ��ı��������︳ֵ��ͬʱ��ʼ������
	��3��forward�� ǰ�򴫲� -> д��unit_forward����
	��4��backward�����򴫲� -> д��unit_fbackward����
	��5��update�� �������� -> д��unit_update����
	��6��load: load������������б�Ҫ����Ҫ����debug�Ͷϵ���ѵ
	��7��nnet_setup.m�� �޸�����ʹ����ʶ�²��type
	
Creat a basic layer: need to write 5+ files:
		(1) xxx_set: set the variable changing in different sampes
		(2) xxx_initial: set the variable unchanging in different samples, meanwhile, the initial the layer params.
		(3) xxx_forward: forward. And write into unit_forward.m, meanwhile.
		(4) xxx_backward: backward. And write into unit_backward.m, meanwhile.
		(5) xxx_update: update the params. And write into unit_update.m, meanwhile.
		(+6) xxx_load: load params, used to debug, and retrain from a cutpoit.
		(7) nnet_setup: modify it in order to known the new layer.
	
	