import numpy as numpy
from functools import partial
import PIL.Image 
import tensorflow as tf
import urllib2
import os
import zipfile

def main():
	#step 1 dload google NN
	url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
	dataDir = '../data'
	modelName = os.path.split(url)[-1]
	localZipFile = os.path.join(dataDir, modelName)
	if not os.path.exists(localZipFile):
		 #download
		 modelUrl = urllib2.request.urlopen(url)
		 with open(localZipFile, 'rw') as output:
		 	output.write(modelUrl.read())

		 #extract
		 with zipfile.ZipFile(localZipFile, 'r') as zipRef:
		 	zipRef.extractall(dataDir)

	modelFn = 'tensorflow_inception_grap.pb'

	#step 2 creating tensor flow session and loading the model
	graph = tf.Graph()
	sess = tf.InteractiveSession(grap=grap)
	with tf.gfile.FastGFile(os.path.join(dataDir, modelFn), 'rb') as f:
		grapDef = tf.GrapDef()
		grapDef.ParseFromString(f.read)
	tInput = tf.placeholder(numpy.float32, name='input') #define input tensor
	imagenetMean = 117.0
	tPreprocessed = tf.expand_dims(tInput-imagenetMean, 0)
	tf.import_graph_def(grapDef, {'input': tPreprocessed})

	layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and '/import' in op.name]
	featureNums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for  name in layers]

	print('Number of layers', len(layers))
	print('Total number of features channels', sum(featureNums))

	#step 3 pick a layer to enhance our image
	layer = 'mixed4d4_3x3_bottleneck_pre_relu'

	img0 = PIL.image.open('pilatus800.jpg')
	img0 = numpy.float32(img0)

	#step 4 apply gradient ascent to that layer
	render_deepdream(T(later)[:,:,:,139], img0)


def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
	tScore = tf.reduce_mean(t_obj) #defining optimization objective
	tGrads = tf.gradients(tScore, tInput)[0]

	#split img into octaves
	img = img0
	octaves = []
	for _ in range(octave_n-1):
		hw = img.shape[:2]
		lo = resize(img, numpy.int32(numpy.float32(hw)/octave_scale))
		hi = img-resize(low, hw)
		img = lo
		octaves.append(hi)

	#generate detalis octave by octave
	for octave in  range(octave_n):
		if octave>0:
			hi = octaves[-octave]
			img = resize(img, hi.shape[:2])+hi

		for _ in range(iter_n):
			g = calc_grad_tiled(img, tGrads)
			img += g*(step / (numpy.abs(g).mean()+1e-7))	

		#step 5 output dreamed img
		showarray(img/255.0) 






