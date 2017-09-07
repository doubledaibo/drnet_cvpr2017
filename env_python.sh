#! /bin/bash
nvidia-docker run --rm -it -e DISPLAY=$DISPLAY -e CAFFE_ROOT=/root/caffe -e PYTHONPATH=/root/caffe/python -p 8886:8888  -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/nlab/masui/shared:/root/shared:Z -w /root/shared/drnet_cvpr2017 proboscis/drnet /usr/bin/python "$@"
#nvidia-docker run --rm -it -e DISPLAY=$DISPLAY -e CAFFE_ROOT=/root/caffe -e PYTHONPATH=/root/caffe/python -p 8886:8888  -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/nlab/masui/shared:/root/shared:Z -w /root/shared/drnet_cvpr2017 --privileged=true proboscis/drnet /usr/bin/python "$@"
