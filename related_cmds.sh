# 生成字典
python 1.generate_dicts.py /dev/torch_nlp/data/rmrb2014/train_corpus.csv 7

#构建语料
python 2.generate_corpus_from_dicts.py  /dev/torch_nlp/data/rmrb2014/train_corpus.csv  /dev/torch_nlp/data/rmrb2014/train_corpus.csv.spdb.word2index.json  /dev/torch_nlp/data/rmrb2014/train_corpus.csv.spdb.tag2index.json 

# 训练脚本 cmds
python line_tag_digital_lstm_crf.py --model-name default --vocab  /dev/torch_nlp/src/bilstm_crf_sequence_labeling_demo/token2idx.json /dev/torch_nlp/src/bilstm_crf_sequence_labeling_demo/tag2idx.json --trainset /dev/torch_nlp/src/bilstm_crf_sequence_labeling_demo/data/train.csv --testset /dev/torch_nlp/src/bilstm_crf_sequence_labeling_demo/data/test.csv


# 启动tensorboard
tensorboard --logdir=./merged_test_1208/ --port 9527

# 构建镜像
docker build -t torch_nlp:0.0.1 .
docker build -f dockerfile.rebuild -t torch_nlp-runnable:v0.0.1 .

# 运行容器
docker run -p 8007:8081 --name robotai-server -d robotai-server:v1.0.0 sh /app/src/start.sh

# 进入镜像
docker run -p 8007:8081 -it --entrypoint /bin/bash robotai-server-runnable:v1.0.1 

# 进入镜像
docker exec -it robotai-server /bin/bash

# 创建容器
docker run -dit --name=belle-tmp 9f01941b6464 /bin/bash

# 保存镜像
docker commit -m '可执行的服务' -a zhangjl19 robotai-server robotai-server-runnable:v1.0.0

# 执行runnable镜像
docker run -p 8007:8081 --name robotai-server -d robotai-server-runnable:v1.0.1 sh /app/src/start.sh

# 导出镜像
docker save -o robotai-server-runnable_v1.0.0.tar robotai-server-runnable:v1.0.0

#加载镜像
docker load -i robotai-server-runnable_v1.0.0.tar

# 打包
tar czvf roboot_ai.tar.gz  --exclude=.git roboot_ai


# 挂载容器
docker run -dit --name=torch_nlp --gpus all --runtime=nvidia -v /etc/localtime:/etc/localtime -v /etc/timezone:/etc/timezone -v /home/zhangjl19/projects/torch_nlp:/dev/torch_nlp -v /home/zhangjl19/data/:/dev/data -v /home/zhangjl19/soft/vscode_related/vscode-server-linux-x64:/root/.vscode-server torch_nlp:0.0.1 /bin/bash
docker run -dit --name=torch_nlp --gpus all --runtime=nvidia -v /etc/localtime:/etc/localtime -v /etc/timezone:/etc/timezone -v /home/zhangjl19/projects:/dev/projects -v /home/zhangjl19/data/:/dev/data torch_nlp:0.0.1 /bin/bash

# 初始化hf
git-lfs install

# 下载bloomz-7b1-mt
git clone https://huggingface.co/bigscience/bloomz-7b1-mt
git clone https://huggingface.co/bigscience/bloomz-1b1

# demo
https://www.wehelpwin.com/article/4008

# 微调 bloomz demo
https://zhuanlan.zhihu.com/p/625488835

# 验证容器中GPU可用
https://cloud.tencent.com/developer/article/1506050

