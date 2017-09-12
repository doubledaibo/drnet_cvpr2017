import caffe
import numpy as np
import json
import math
import os.path as osp
import os
import cv2
import h5py

class RelDataLayer(caffe.Layer):
	def _shuffle_inds(self):
		self._perm = np.random.permutation(np.arange(self._num_instance))
		self._cur = 0

	def _get_next_batch_ids(self):
		if self._cur + self._batch_size > self._num_instance:	
			self._shuffle_inds()
		ids = self._perm[self._cur : self._cur + self._batch_size]
		self._cur += self._batch_size	
		return ids

	def _getAppr(self, im, bb):
		subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
		subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
		pixel_means = np.array([[[103.939, 116.779, 123.68]]])
		subim -= pixel_means
		subim = subim.transpose((2, 0, 1))
		return subim

	def _getDualMask(self, ih, iw, bb):
		rh = 32.0 / ih
		rw = 32.0 / iw
		x1 = max(0, int(math.floor(bb[0] * rw)))
		x2 = min(32, int(math.ceil(bb[2] * rw)))
		y1 = max(0, int(math.floor(bb[1] * rh)))
		y2 = min(32, int(math.ceil(bb[3] * rh)))
		mask = np.zeros((32, 32))
		mask[y1 : y2, x1 : x2] = 1
		assert(mask.sum() == (y2 - y1) * (x2 - x1))
		return mask		
	
	def _get_next_batch(self):
		ids = self._get_next_batch_ids()
		qas = []
		qbs = []
		ims = []
		poses = []
		labels = []
		for id in ids:
			sample = self._samples[id]
			im = cv2.imread(sample["imPath"]).astype(np.float32, copy=False)
			ih = im.shape[0]	
			iw = im.shape[1]
			qa = np.zeros(self._nclass)
			qa[sample["aLabel"] - 1] = 1
			qas.append(qa)
			qb = np.zeros(self._nclass)
			qb[sample["bLabel"] - 1] = 1
			qbs.append(qb)
			ims.append(self._getAppr(im, sample["rBBox"]))
			poses.append([self._getDualMask(ih, iw, sample["aBBox"]), \
					self._getDualMask(ih, iw, sample["bBBox"])])
			labels.append(sample["rLabel"])
		return {"qa": np.array(qas), "qb": np.array(qbs), "im": np.array(ims), "posdata": np.array(poses), "labels": np.array(labels)}
	
	def setup(self, bottom, top):
		layer_params = json.loads(self.param_str)
		self._samples = json.load(open(layer_params["dataset"]))
		self._num_instance = len(self._samples)
		self._batch_size = layer_params["batch_size"]
		self._nclass = layer_params["nclass"]
		
		self._name_to_top_map = {"qa": 0, "qb": 1, "im": 2, "posdata": 3, "labels": 4}
		self._shuffle_inds()
		top[0].reshape(self._batch_size, self._nclass)
		top[1].reshape(self._batch_size, self._nclass)
		top[2].reshape(self._batch_size, 3, 224, 224)
		top[3].reshape(self._batch_size, 2, 32, 32)
		top[4].reshape(self._batch_size)
		
	def forward(self, bottom, top):
		batch = self._get_next_batch()
		for blob_name, blob in batch.iteritems():
			idx = self._name_to_top_map[blob_name]	
			top[idx].reshape(*(blob.shape))
			top[idx].data[...] = blob.astype(np.float32, copy=False)
	
	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass

