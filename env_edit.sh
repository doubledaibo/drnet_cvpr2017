#! /bin/bash
nvidia-docker run --rm -it -e DISPLAY=$DISPLAY -p 8886:8888  -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/nlab/masui/shared:/root/shared -w /root/shared/drnet_cvpr2017 proboscis/drnet
