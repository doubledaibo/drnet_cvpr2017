#! /bin/bash
nvidia-docker run --rm -it -e DISPLAY=$DISPLAY -e CAFFE_ROOT=/root/caffe -e PYTHONPATH=/root/caffe/python -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/nlab/masui/shared:/root/shared -w /root/shared/drnet_cvpr2017 proboscis/drnet:faster-rcnn "$@"

