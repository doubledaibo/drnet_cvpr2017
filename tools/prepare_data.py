import os.path as osp
import mat4py as mp
import cv2
import numpy as np
import math
import json


class DataLoader:
    def __init__(self, datasetRoot, split):
        self._root = datasetRoot
        self._split = split
        self.loadLabels()
        self.loadAnnotation(split)

    def loadLabels(self):
        mat = mp.loadmat(osp.join(self._root, "predicate.mat"))
        self.relList = mat["predicate"]
        self.numRelClass = len(self.relList)
        self.relMapping = {}
        for i in xrange(len(self.relList)):
            self.relMapping[self.relList[i]] = i
        mat = mp.loadmat(osp.join(self._root, "objectListN.mat"))
        self.objList = mat["objectListN"]
        self.numObjClass = len(self.objList) + 1
        self.objMapping = {}
        self.objMapping["__BG"] = 0
        for i in xrange(len(self.objList)):
            self.objMapping[self.objList[i]] = i + 1

    def loadAnnotation(self, split):
        mat = mp.loadmat(osp.join(self._root, "annotation_" + split + ".mat"))
        self.annotations = mat["annotation_" + split]

    def getNumImgs(self):
        return len(self.annotations)

    def getImPath(self, idx):
        return osp.join(self._root, "images", self._split, self.annotations[idx]["filename"])

    def getNumRel(self):
        numRels = 0
        n = self.getNumImgs()
        for i in xrange(n):
            rels = self.getRels(i)
            numRels += len(rels)
        return numRels

    def getRels(self, idx):
        if "relationship" in self.annotations[idx]:
            rels = self.annotations[idx]["relationship"]
            if isinstance(rels, dict):
                rels = [rels]
            return rels
        else:
            return []

    def outputDB(self, type, data):
        json.dump(data, open(type + self._split + ".json", "w"))

    def bboxTransform(self, bbox, ih, iw):  # [x1, y1, x2, y2]
        return [max(bbox[2], 0), max(bbox[0], 0), min(bbox[3] + 1, iw), min(bbox[1] + 1, ih)]

    def getRelLabel(self, predicate):
        if not (predicate in self.relMapping):
            return -1
        return self.relMapping[predicate]

    def getObjLabel(self, predicate):
        if not (predicate in self.objMapping):
            return -1
        return self.objMapping[predicate]

    def getUnionBBox(self, aBB, bBB, ih, iw, margin=10):
        return [max(0, min(aBB[0], bBB[0]) - margin),
                max(0, min(aBB[1], bBB[1]) - margin),
                min(iw, max(aBB[2], bBB[2]) + margin),
                min(ih, max(aBB[3], bBB[3]) + margin)]

    def getRelSamplesSingle(self):
        n = self.getNumImgs()
        self._sampleIdx = 0
        samples = []
        for i in xrange(n):
            rels = self.getRels(i)
            if len(rels) == 0:
                continue

            path = self.getImPath(i)
            im = cv2.imread(path)
            ih = im.shape[0]
            iw = im.shape[1]
            for rel in rels:
                phrase = rel["phrase"]
                rLabel = self.getRelLabel(phrase[1])
                aLabel = self.getObjLabel(phrase[0])
                bLabel = self.getObjLabel(phrase[2])
                aBBox = self.bboxTransform(rel["subBox"], ih, iw)
                bBBox = self.bboxTransform(rel["objBox"], ih, iw)
                rBBox = self.getUnionBBox(aBBox, bBBox, ih, iw)
                samples.append({"imPath": path, "rLabel": rLabel, "aLabel": aLabel, "bLabel": bLabel, "rBBox": rBBox,
                                "aBBox": aBBox, "bBBox": bBBox})
                self._sampleIdx += 1
                if self._sampleIdx % 100 == 0:
                    print self._sampleIdx
        self.outputDB("rel", samples)

    def image_paths(self):
        return [self.getImPath(i) for i in xrange(self.getNumImgs())]




if __name__ == "__main__":
    loader = DataLoader("/root/shared/drnet_cvpr2017/dataset", "test")
    #loader._getRelSamplesSingle()

    # TODO make image_paths
    # image_paths file format: [ path of image_i for image_i in images ]
    # TODO make obj_dets_file
    # obj_dets_file format: [ obj_dets of image_i for image_i in images ]
    # 	obj_dets: numpy.array of size: num_instance x 5
    # 		instance: [x1, y1, x2, y2, prob, label]
