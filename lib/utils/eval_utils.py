def computeArea(bb):
	return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

def computeIoU(bb1, bb2):
	ibb = [max(bb1[0], bb2[0]), \
		max(bb1[1], bb2[1]), \
		min(bb1[2], bb2[2]), \
		min(bb1[3], bb2[3])]
	iArea = computeArea(ibb)
	uArea = computeArea(bb1) + computeArea(bb2) - iArea
	return (iArea + 0.0) / uArea

