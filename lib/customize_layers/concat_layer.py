import caffe
import numpy as np
import json

class Layer(caffe.Layer):
	def setup(self, bottom, top):
		self._sum = 0
		for i in xrange(len(bottom)):
			self._sum += bottom[i].data.shape[1]

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].data.shape[0], self._sum)

	def forward(self, bottom, top):
		offset = 0
		for i in xrange(len(bottom)):
			if len(bottom[i].data.shape) == 2:
				top[0].data[:, offset : offset + bottom[i].data.shape[1]] = bottom[i].data
			else:
				top[0].data[:, offset : offset + bottom[i].data.shape[1]] = bottom[i].data[:, :, 0, 0]
			offset += bottom[i].data.shape[1]
	
	def backward(self, top, propagate_down, bottom):
		offset = 0
		for i in xrange(len(bottom)):
			if len(bottom[i].data.shape) == 2:
				bottom[i].diff[...] = top[0].diff[:, offset : offset + bottom[i].data.shape[1]]
			else:
				bottom[i].diff[:, :, 0, 0] = top[0].diff[:, offset : offset + bottom[i].data.shape[1]]
			offset += bottom[i].data.shape[1]

		
