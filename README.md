环境配置的话目前是python3.12, 其他的用的是SAM2中的环境配置，https://github.com/facebookresearch/sam2. clone下来pip install -e . 即可。

训练代码就是train.py 和util.py,调参数主要是在train中的基本训练参数，还有util中loss的比例参数（k, alpha, beta, gamma），可以初始化加载权重，进行微调。

predict.py是对已经有的模型权重进行测试。通过存储在文件中进行保存，浏览。直接python 文件命令行即可。
compare_source.py是对同一mask，不同source的测试，由于没有ground truth，目前就只展示结果。

dataExtract.py是对原始数据进行处理。
dataset.py沿用lithbench，进行部分修改，将图片组成可用数据集。

模型架构imageEncoder与sam一致，sourceEncoder根据论文进行修改。 maskDecoer根据任务特性进行了一定修改。

数据不多，目前没加入分布式训练，一张GPU即可，已有参数是针对80GB显存的。