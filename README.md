# Rl

# 解决rending 导入问题
cp lib/rendering.py /home/zhangjl19/miniconda3/envs/rl_gy/lib/python3.8/site-packages/gym/envs/classic_control/
在 classic_control的init 文件中增加一行 from gym.envs.classic_control import rendering

# 参考资料
> https://blog.51cto.com/u_14622170/5266593 # rending 导入错误


## Getting started