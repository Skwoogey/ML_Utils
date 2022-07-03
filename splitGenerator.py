import tensorflow as tf
import numpy as np
import math
import random

class splitGenerator:
	def __init__(self, data_per_class):
		self.data = data_per_class
		self.num_of_classes = len(self.data)
		self.available_indecies = []
		self.sample_numbers = []
		for class_i in range(len(self.data)):
			cur_class = self.data[class_i]
			self.sample_numbers.append(cur_class.shape[0])
			self.available_indecies.append(list(range(cur_class.shape[0])))
			random.shuffle(self.available_indecies[-1])

	def refillClass(self, class_i):
		self.available_indecies[class_i] = list(range(self.sample_numbers[class_i]))
		random.shuffle(self.available_indecies[class_i])

	def getSamplesFromClass(self, class_i, samples_num):
		if samples_num <= len(self.available_indecies[class_i]):
			chosen_indecies = self.available_indecies[class_i][:samples_num]
			self.available_indecies[class_i] = self.available_indecies[class_i][samples_num:]
			return self.data[class_i][chosen_indecies]
		else:
			indecies_num = len(self.available_indecies[class_i])
			chosen_indecies = self.available_indecies[class_i]
			self.refillClass(class_i)
			list_i = 0
			while len(chosen_indecies) != samples_num:
				if self.available_indecies[class_i][list_i] not in chosen_indecies:
					chosen_indecies.append(self.available_indecies[class_i].pop(list_i))
				list_i += 1
			return self.data[class_i][chosen_indecies]

	def getFullSplit(self, class_sample_nums, augment = False):
		samples = []
		for class_i in range(self.num_of_classes):
			samples.append(self.getSamplesFromClass(class_i, class_sample_nums[class_i]))

		samples = np.concatenate(samples, axis=0)
		labels = []
		for class_i in range(self.num_of_classes):
			labels = labels + [class_i] * class_sample_nums[class_i]
		if augment:
			augmented_samples = []
			for sample in samples:
				augmented_samples.append(np.rot90(sample, 1))
				augmented_samples.append(np.rot90(sample, 2))
				augmented_samples.append(np.rot90(sample, 3))
				mirrored_sample = np.flip(sample, axis = 0)
				augmented_samples.append(mirrored_sample)
				augmented_samples.append(np.rot90(mirrored_sample, 1))
				augmented_samples.append(np.rot90(mirrored_sample, 2))
				augmented_samples.append(np.rot90(mirrored_sample, 3))


			augmented_samples = np.stack(augmented_samples, axis = 0)
			samples = np.concatenate((samples, augmented_samples), axis = 0)

			for class_i in range(self.num_of_classes):
				labels = labels + [class_i] * class_sample_nums[class_i] * 7

		labels = np.array(labels)
		samples = samples.reshape((*samples.shape, 1))
		labels = labels.reshape((*labels.shape, 1))
		return samples, labels

	def getRemainingSamples(self, augment = False):
		return self.getFullSplit([len(x) for x in self.available_indecies], augment)

	def getPercentageSplit(self, percentage, augment = False):
		return self.getFullSplit([int(x * percentage) for x in self.sample_numbers], augment)

	def getCountSplit(self, count, augment = False):
		return self.getFullSplit([count] * self.num_of_classes, augment)

		
