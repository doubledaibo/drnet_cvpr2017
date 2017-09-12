#!/usr/bin/env python

import _init_paths
import caffe
import argparse
import time, os, sys
import json
import cv2
import cPickle as cp
import numpy as np
import math


def parse_args():
    """
	Parse input arguments
	"""
    parser = argparse.ArgumentParser()
    # image_paths file format: [ path of image_i for image_i in images ]
    # the order of images in image_paths should be the same with obj_dets_file
    parser.add_argument('--image_paths', dest='image_paths', help='file containing test dataset',
                        default='', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--num_dets', dest='max_det',
                        help='max number of detections per image',
                        default=100, type=int)
    # obj_dets_file format: [ obj_dets of image_i for image_i in images ]
    # 	obj_dets: numpy.array of size: num_instance x 5
    # 		instance: [x1, y1, x2, y2, prob, label]
    parser.add_argument('--obj_dets_file', dest='obj_dets_file',
                        help='file containing object detections',
                        default=None, type=str)
    # type 0: im only --input
    # type 1: pos only
    # type 2: im + pos
    # type 3: im + pos + qa + qb
    parser.add_argument('--input_type', dest='type',
                        help='type of input sets',
                        default=0, type=int)

    parser.add_argument('--ncls', dest='num_class', help='number of object classes', default=101, type=int)

    parser.add_argument('--out', dest='out', help='name of output file', default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def getPred(pred, max_num_det):
    """return top max_num_det regions"""
    if pred.shape[0] == 0:
        return pred
    inds = np.argsort(pred[:, 4])
    inds = inds[::-1]
    if len(inds) > max_num_det:
        inds = inds[:max_num_det]
    return pred[inds, :]


def getUnionBBox(aBB, bBB, ih, iw):
    margin = 10
    return [max(0, min(aBB[0], bBB[0]) - margin),
            max(0, min(aBB[1], bBB[1]) - margin),
            min(iw, max(aBB[2], bBB[2]) + margin),
            min(ih, max(aBB[3], bBB[3]) + margin)]


def getAppr(im, bb):
    subim = im[bb[1]: bb[3], bb[0]: bb[2], :]
    subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0],
                       interpolation=cv2.INTER_LINEAR)
    pixel_means = np.array([[[103.939, 116.779, 123.68]]])
    subim -= pixel_means
    subim = subim.transpose((2, 0, 1))
    return subim


def getDualMask(ih, iw, bb):
    rh = 32.0 / ih
    rw = 32.0 / iw
    x1 = max(0, int(math.floor(bb[0] * rw)))
    x2 = min(32, int(math.ceil(bb[2] * rw)))
    y1 = max(0, int(math.floor(bb[1] * rh)))
    y2 = min(32, int(math.ceil(bb[3] * rh)))
    mask = np.zeros((32, 32))
    mask[y1: y2, x1: x2] = 1
    assert (mask.sum() == (y2 - y1) * (x2 - x1))
    return mask


def forward_batch(net, ims, poses, qas, qbs, args):
    forward_args = {}
    if args.type != 1:
        net.blobs["im"].reshape(*(ims.shape))
        forward_args["im"] = ims.astype(np.float32, copy=False)
    if args.type != 0:
        net.blobs["posdata"].reshape(*(poses.shape))
        forward_args["posdata"] = poses.astype(np.float32, copy=False)
    if args.type == 3:
        net.blobs["qa"].reshape(*(qas.shape))
        forward_args["qa"] = qas.astype(np.float32, copy=False)
        net.blobs["qb"].reshape(*(qbs.shape))
        forward_args["qb"] = qbs.astype(np.float32, copy=False)
    net_out = net.forward(**forward_args)
    itr_pred = net_out["pred"].copy()
    return itr_pred


def test_net(net, image_paths, args):
    f = open(args.obj_dets_file, "r")
    all_dets = cp.load(f)
    f.close()
    num_img = len(image_paths)
    num_class = args.num_class
    thresh = 0.05
    max_num_det = args.max_det
    batch_size = 30
    pred = []
    pred_bboxes = []
    for i in xrange(num_img):
        im = cv2.imread(image_paths[i]).astype(np.float32, copy=False)
        ih = im.shape[0]
        iw = im.shape[1]

        dets = getPred(all_dets[i], max_num_det)  # top max_num_det detected regions
        num_dets = dets.shape[0]
        pred.append([])
        pred_bboxes.append([])
        for subIdx in xrange(num_dets):
            ims = []
            poses = []
            qas = []
            qbs = []
            for objIdx in xrange(num_dets): #for each subj and obj
                if subIdx != objIdx:
                    sub = dets[subIdx, 0: 4]
                    obj = dets[objIdx, 0: 4]
                    rBB = getUnionBBox(sub, obj, ih, iw)
                    rAppr = getAppr(im, rBB)
                    rMask = np.array([getDualMask(ih, iw, sub), getDualMask(ih, iw, obj)])
                    ims.append(rAppr)
                    poses.append(rMask)
                    qa = np.zeros(num_class - 1)
                    qa[dets[subIdx, 5] - 1] = 1
                    qb = np.zeros(num_class - 1)
                    qb[dets[objIdx, 5] - 1] = 1
                    qas.append(qa)
                    qbs.append(qb)
            if len(ims) == 0:
                break
            ims = np.array(ims)
            poses = np.array(poses)
            qas = np.array(qas)
            qbs = np.array(qbs)
            _cursor = 0
            itr_pred = None
            num_ins = ims.shape[0]
            while _cursor < num_ins:
                _end_batch = min(_cursor + batch_size, num_ins)
                # I guess this includes the calculation of pair-filter
                itr_pred_batch = forward_batch(net, ims[_cursor: _end_batch] if ims.shape[0] > 0 else None,
                                               poses[_cursor: _end_batch] if poses.shape[0] > 0 else None,
                                               qas[_cursor: _end_batch] if qas.shape[0] > 0 else None,
                                               qbs[_cursor: _end_batch] if qbs.shape[0] > 0 else None, args)
                if itr_pred is None:
                    itr_pred = itr_pred_batch
                else:
                    itr_pred = np.vstack((itr_pred, itr_pred_batch))
                _cursor = _end_batch

            cur = 0
            for objIdx in xrange(num_dets):  # for each subj,obj
                if subIdx != objIdx:
                    sub = dets[subIdx, 0: 4]
                    obj = dets[objIdx, 0: 4]
                    for j in xrange(itr_pred.shape[1]):
                        if itr_pred[cur, j] < thresh:  # pair filter? no, this may be a consciousness
                            continue
                        pred[i].append(
                            [itr_pred[cur, j], dets[subIdx, 4], dets[objIdx, 4], dets[subIdx, 5], j, dets[objIdx, 5]])
                        pred_bboxes[i].append([sub, obj])
                    cur += 1
            assert (cur == itr_pred.shape[0])
        pred[i] = np.array(pred[i])
        pred_bboxes[i] = np.array(pred_bboxes[i])
    print "writing file.."
    f = open(args.out, "wb")
    cp.dump([pred, pred_bboxes], f, cp.HIGHEST_PROTOCOL)
    f.close()


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    test_image_paths = json.load(open(args.image_paths))

    test_net(net, test_image_paths, args)
