# Code of [Detecting Visual Relationship with Deep Relational Networks](https://arxiv.org/abs/1704.03114)

The code is written in python, and all networks are implemented using [Caffe](https://github.com/BVLC/caffe).

## Datasets 

* [VRD](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)
* sVG: subset of [Visual Genome](https://visualgenome.org/)
will be available soon.

## Networks

This repo contains three kinds of networks. And all of them get the raw response for predicate based on both appearance cues and spatial cues,
followed by a refinement according to responses of the subject, the object and the predicate.
The networks are designed for the task of predicate recognition, 
where ground-truth labels of the subject and the object are provided as inputs.
Therefore, in these networks, responses of the subject and the object are replaced with indicator vectors,
and only response of the predicate will be refined.

In these networks, the subnet for appearance cues is VGG16, and the subnet for spatial cues consists of three conv layers.
And outputs of both subnets are combined via a customized concatenate layer,
followed by two fc layers to generate raw response for the predicate.

The customized concatenate layer is used for combining the output of a fc layer and channels of the output of a conv layer,
which can be replaced with caffe's Concat layer
if the last conv layer in spatial subnet (conv3_p) is equivalently replaced with a fc layer.

The details of these networks are

* drnet_8units_softmax: it has 8 inference units with softmax function as the activation function.

* drnet_8units_linear_shareweight: it has 8 inference units with no activation function, and the weights are shared across units.

* drnet_8units_relu_shareweight: it has 8 inference units with relu function as the activation function, and the weights are shared across units.

### Recalls on Predicate Recognition

| Networks | Recall@50 | Recall@100 |
| --- | :---: | :---: |
| drnet_8units_softmax | 77.01 | 78.28 |
| drnet_8units_linear_shareweight | 80.66 | 81.85 |
| drnet_8units_relu_shareweight | 82.52 | 83.71 |

## Codes

* lib/: python layers, as well as auxiliary files for evaluation
* prototxts/: training and testing prototxts
* tools/: python codes for preparing data and evaluation
* snapshots/: pretrain models

## Finetune or Evaluate

1. Download the dataset [VRD](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)
2. Preprocess the dataset using tools/preprare_data.py
3. Download one pretrain model in snapshots/
4. Finetune or Evaluate using corresponding prototxts in prototxts/

## Citation

If you use this code, please cite the following paper(s):

	@article{dai2017detecting,
		title={Detecting Visual Relationships with Deep Relational Networks},
		author={Dai, Bo and Zhang, Yuqi and Lin, Dahua},
  		journal={arXiv preprint arXiv:1704.03114},
  		year={2017}
	}

