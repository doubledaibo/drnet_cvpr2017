#! /bin/bash
nvidia-docker run --name=drnet-edit --rm -it -e DISPLAY=$DISPLAY -e CAFFE_ROOT=/root/caffe -e PYTHONPATH=/root/caffe/python -p 8886:8888  -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/nlab/masui/shared:/root/shared -w /root/shared/drnet_cvpr2017 proboscis/drnet
