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

# 创建conda 环境
conda create --name rl_gy python=3.8

# 激活环境
conda activate rl_gy

# 取消环境
conda deactivate


# 增加 rending 渲染
cp lib/rendering.py /home/zhangjl19/miniconda3/envs/rl_gy/lib/python3.8/site-packages/gym/envs/classic_control/
在 classic_control的init 文件中增加一行 from gym.envs.classic_control import rendering