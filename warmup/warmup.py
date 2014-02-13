#Neel Patel
#npatel@college.harvard.edu
#Practical 1 Warmup

import numpy as np
import copy
import matplotlib.pyplot as plt
import Image
import math
from collections import Counter
import time 
from scipy.cluster.vq import *

#provided function to unzip the given CIFAR data (taken from CIFAR site)
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#my own distance function: it may be faster to use Numpy LinAlg methods
def distance(img1, img2):
	return np.sqrt(np.sum(np.square(img1-img2)))

#implementation of k-means
def kmeans(data, k, max=10):
	starttime = time.time()
	size = len(data)
	width = len(data[0])
	r = np.zeros((size, k))
	random = np.random.randint(0,k,size)
	means = np.zeros((k, width))

	#randomly initialize one zero in each
	for i in range(size):
		r[i][random[i]] = 1		

	unchanged = False
	iterations = 0

	while not unchanged:
		iterations = iterations + 1		

		r_copy = copy.deepcopy(r)

		sums_in_k = np.sum(r, axis=0)

		#zeroes everything
		for i in range(k):			
			total = np.zeros(width)
			means[i] = total

		#for each element, adds the data to the relevant total
		for j in range(size):
			index = np.argmax(r[j])
			means[index] = np.add(means[index],data[j])

		#calculates totals by dividing
		for i in range(k):	
			#saves the new total
			means[i] = 1.0 / sums_in_k[i] * means[i]

		#print "Got means: " + str(time.time() - starttime)

		for i in range(size):
			r[i] = np.zeros(k)

			#finds the closest mean
			distances = np.zeros(k)
			for j in range(k):
				#distances[j] = distance(data[i], means[j])
				diff = np.subtract(data[i], means[j])
				
				distances[j] = np.dot(diff, diff)
			r[i][np.argmin(distances)] = 1

		#print "Got mins: " + str(time.time() - starttime)

		if np.array_equal(r_copy, r) or iterations > max:
			unchanged = True
	return (r,means)

def otherkmeans(data, k):
	means, labels = kmeans2(data,k)
	return (means, labels)

#displays the image
def display(data):	
	#first packages the data into triplets
	rgb = zip(data[0:1024], data[1024:2048], data[2048:])
	
	#reshapes it so it can be displayed
	rgb_arr = np.array(rgb).reshape(32,32,3)
		
	img = Image.fromarray(rgb_arr, 'RGB')
	return img

#displays a grid of images
def displayGrid(data_array):
	count = len(data_array)

	#sets grid gutter
	gutter = 0

	#we'll display 10 images horizontally
	width = int((32 + gutter) * 10)
	height = int(math.ceil(count / 10.0) * (32 + gutter))
	
	blank = np.zeros( (height,width,3), dtype=np.uint8)
	grid = Image.fromarray(blank, 'RGB')

	x = 0
	y = 0
	for current in data_array:
		if (x + 32 + gutter) > width:
			x = 0
			y = y + (32 + gutter)

		grid.paste(display(current), (x,y))
		x = x + (32 + gutter)

	grid.show()

#given a k-by-k matrix comparing our categorization to scipy's, prints a comparison
def score(comparison):
	for i in range(len(comparison)):
		print str(i) + ": " + str(float(np.max(comparison[i])) / float(np.sum(comparison[i])))

		
def main():
	prefix = "cifar-10-batches-py/"
	file_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

	#loads each file into a dictionary	

	images = []
	labels = []

	for name in file_names[:1]:
		result = unpickle(prefix + name)
		images.extend(result["data"])
		labels.extend(result["labels"])	

	images = np.array(images)

	k = 5	

	starttime = time.time()	
	r, means = kmeans(images,k, 10)
	print str(time.time() - starttime) + " sec for my k-means"

	starttime = time.time()
	scipy_means, scipy_labels = otherkmeans(images, k)	
	print str(time.time() - starttime) + " sec for SciPy's k-means"

	results = [[] for _ in range(k)]
	categories = [[] for _ in range(k)]

	comparison = [[ 0 for _ in range(k)] for _ in range(k)]

	#writes the data on comparing my classification to the scipy classification
	output = open('data.txt', 'w')
	for i in range(len(r)):		
		index = np.argmax(r[i])
		results[index].append(images[i])
		categories[index].append(labels[i])
		output.write(str(i) + "\t" + str(scipy_labels[i]) + "\t" + str(index) + "\n")
		comparison[index][scipy_labels[i]] = comparison[index][scipy_labels[i]] + 1
	
	output.close()

	score(comparison)

	for row in results:		
		displayGrid(row[:500])

main()