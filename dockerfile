#使用的基础镜像，构建时本地没有的话会先下载
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

LABEL maintainer="zhangjl19@spdb.com.cn"
LABEL version = "0.0.1"
LABEL description = "torch gpu envs"

WORKDIR /dev
COPY requirements.txt /tmp/requirements.txt
COPY /etc/localtime /etc/localtime

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    #ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo 'Asia/Shanghai' >/etc/timezone