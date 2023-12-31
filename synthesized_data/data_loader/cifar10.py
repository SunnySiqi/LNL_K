import sys
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import json
import os
import copy

def fix_seed(seed=888):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

def get_cifar10(root, cfg_trainer, train=True,
				transform_train=None, transform_val=None,
				download=False, noise_file = '', teacher_idx=None, seed=888):
	base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
	if cfg_trainer['asym']:
		if train:
			fix_seed(seed)
			train_idxs, val_idxs = train_val_split(base_dataset.targets, seed)
			train_dataset = CIFAR10_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, seed=seed)
			val_dataset = CIFAR10_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
			train_dataset.asymmetric_noise()
			# val_dataset.asymmetric_noise()
				  
			if teacher_idx is not None:
				print(len(teacher_idx))
				train_dataset.truncate(teacher_idx)
			print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
		else:
			fix_seed(seed)
			train_dataset = []
			val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train, transform=transform_val)
			print(f"Test: {len(val_dataset)}")

		return train_dataset, val_dataset
	elif cfg_trainer['control']:
		if train:
			fix_seed(seed)
			train_idxs, val_idxs = train_val_split_ctl(base_dataset.targets, cfg_trainer, seed)
			train_dataset = CIFAR10_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, seed=seed)
			val_dataset = CIFAR10_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
			train_dataset.control_noise()
			#val_dataset.control_noise()
				  
			if teacher_idx is not None:
				print(len(teacher_idx))
				train_dataset.truncate(teacher_idx)
			print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
		else:
			fix_seed(seed)
			train_dataset = []
			val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train, transform=transform_val)
			print(f"Test: {len(val_dataset)}")

		return train_dataset, val_dataset


def train_val_split(base_dataset: torchvision.datasets.CIFAR10, seed):
	fix_seed(seed)
	num_classes = 10
	base_dataset = np.array(base_dataset)
	train_n = int(len(base_dataset) * 0.9 / num_classes)
	train_idxs = []
	val_idxs = []

	for i in range(num_classes):
		idxs = np.where(base_dataset == i)[0]
		np.random.shuffle(idxs)
		train_idxs.extend(idxs[:train_n])
		val_idxs.extend(idxs[train_n:])
	np.random.shuffle(train_idxs)
	np.random.shuffle(val_idxs)

	return train_idxs, val_idxs

def train_val_split_ctl(base_dataset: torchvision.datasets.CIFAR10, cfg_trainer, seed):
	fix_seed(seed)
	num_classes = 10
	base_dataset = np.array(base_dataset)
	ctl_cls_lower_bound = 5
	non_ctl_train_n = int(cfg_trainer['non_ctl_cls_sample']*0.9)
	ctl_train_n = int(cfg_trainer['ctl_cls_sample']*0.9)
	non_ctl_val_n = int(cfg_trainer['non_ctl_cls_sample']*0.1)
	ctl_val_n = int(cfg_trainer['ctl_cls_sample']*0.1)
	train_idxs = []
	val_idxs = []

	for i in range(ctl_cls_lower_bound):
			idxs = np.where(base_dataset == i)[0]
			np.random.shuffle(idxs)
			train_idxs.extend(idxs[:non_ctl_train_n])
			left_idx = idxs[non_ctl_train_n:]
			np.random.shuffle(left_idx)
			val_idxs.extend(left_idx[:non_ctl_val_n])

	for i in range(ctl_cls_lower_bound, num_classes):
			idxs = np.where(base_dataset == i)[0]
			np.random.shuffle(idxs)
			train_idxs.extend(idxs[:ctl_train_n])
			left_idx = idxs[ctl_train_n:]
			np.random.shuffle(left_idx)
			val_idxs.extend(left_idx[:ctl_val_n])

	np.random.shuffle(train_idxs)
	np.random.shuffle(val_idxs)

	return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):
	def __init__(self, root, cfg_trainer, indexs, train=True,
				 transform=None, target_transform=None,
				 download=False, seed=888):
		super(CIFAR10_train, self).__init__(root, train=train,
											transform=transform, target_transform=target_transform,
											download=download)
		fix_seed(seed)
		self.num_classes = 10
		self.cfg_trainer = cfg_trainer
		self.train_data = self.data[indexs]#self.train_data[indexs]
		self.train_labels = np.array(self.targets)[indexs]#np.array(self.train_labels)[indexs]
		self.indexs = indexs
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()
		self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
		self.noise_indx = []
		self.seed = seed
		
	def symmetric_noise(self):
		self.train_labels_gt = self.train_labels.copy()
		fix_seed(self.seed)
		indices = np.random.permutation(len(self.train_data))
		for i, idx in enumerate(indices):
			if i < self.cfg_trainer['percent'] * len(self.train_data):
				self.noise_indx.append(idx)
				self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()

	def control_noise(self):
		self.whole_train_labels_gt = self.train_labels.copy()
		control_lower_idx = 5
		num_ctl_cls = 5
		num_mislabel_per_ctl = int((0.9*self.cfg_trainer['ctl_cls_sample'] - 0.9*self.cfg_trainer['non_ctl_cls_sample'])/2)
		print(num_mislabel_per_ctl)
		for i in range(5, 10):
				idxs = np.where((self.train_labels == i) == True)[0]
				np.random.shuffle(idxs)
				for j in range(num_mislabel_per_ctl):
						self.train_labels[idxs[j]] = np.random.randint(0, 5)
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()
		self.train_labels_gt = self.whole_train_labels_gt.copy()

	def asymmetric_noise(self):
		fix_seed(self.seed)
		self.whole_train_labels_gt = self.train_labels.copy()
		for i in range(self.num_classes):
			indices = np.where(self.train_labels == i)[0]
			np.random.shuffle(indices)
			for j, idx in enumerate(indices):
				if j < self.cfg_trainer['percent'] * len(indices):
					self.noise_indx.append(idx)
					# truck -> automobile
					if i == 9:
						self.train_labels[idx] = 1
					# automobile -> truck
					# elif i == 1:
					# 	self.train_labels[idx] = 9
					# cat -> dog
					elif i == 3:
						self.train_labels[idx] = 5
					# dog -> cat
					# elif i == 5:
					# 	self.train_labels[idx] = 3
					# deer -> horse
					elif i == 4:
						self.train_labels[idx] = 7
					# horse -> deer
					# elif i == 7:
					# 	self.train_labels[idx] = 4
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()
		self.train_labels_gt = self.whole_train_labels_gt.copy()
				
	def switch_data(self):
		self.train_data = self.whole_train_data.copy()
		self.train_labels = self.whole_train_labels.copy()
		self.train_labels_gt = self.whole_train_labels_gt.copy()

	def adjust_base_indx_tmp(self, idx):
		new_train_data = np.array(self.train_data)[idx]
		new_train_labels = np.array(self.train_labels)[idx]
		new_train_labels_gt = np.array(self.train_labels_gt)[idx]
		self.train_data = new_train_data
		self.train_labels = new_train_labels
		self.train_labels_gt = new_train_labels_gt

	def truncate(self, teacher_idx):
		self.train_data = self.train_data[teacher_idx]
		self.train_labels = self.train_labels[teacher_idx]
		self.train_labels_gt = self.train_labels_gt[teacher_idx]

	def train_balance_sample(self):
		ssets = []
		for i in range(0, 10):
			idxs = np.where((self.train_labels == i) == True)[0]
			total_class = int((0.9*self.cfg_trainer['ctl_cls_sample'] - 0.9*self.cfg_trainer['non_ctl_cls_sample'])/2 + 0.9*self.cfg_trainer['non_ctl_cls_sample'])
			need_to_add = total_class - len(list(idxs))
			if need_to_add > 0:
				sub_t = np.random.choice(idxs, int(need_to_add))
				ssets.append(sub_t)
			ssets.append(idxs)
		ssets = np.concatenate(ssets, axis=0)
		self.adjust_base_indx_tmp(ssets)

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)


		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img,target, index, target_gt

	def __len__(self):
		return len(self.train_data)



class CIFAR10_val(torchvision.datasets.CIFAR10):

	def __init__(self, root, cfg_trainer, indexs, train=True,
				 transform=None, target_transform=None,
				 download=False):
		super(CIFAR10_val, self).__init__(root, train=train,
										  transform=transform, target_transform=target_transform,
										  download=download)

		# self.train_data = self.data[indexs]
		# self.train_labels = np.array(self.targets)[indexs]
		self.num_classes = 10
		self.cfg_trainer = cfg_trainer
		if train:
			self.train_data = self.data[indexs]
			self.train_labels = np.array(self.targets)[indexs]
		else:
			self.train_data = self.data
			self.train_labels = np.array(self.targets)
		self.train_labels_gt = self.train_labels.copy()
	def symmetric_noise(self):
		
		indices = np.random.permutation(len(self.train_data))
		for i, idx in enumerate(indices):
			if i < self.cfg_trainer['percent'] * len(self.train_data):
				self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

	def control_noise(self):
		self.train_labels_gt = self.train_labels.copy()
		control_lower_idx = 5
		num_ctl_cls = 5
		num_mislabel_per_ctl = int((0.1*self.cfg_trainer['ctl_cls_sample'] - 0.1*self.cfg_trainer['non_ctl_cls_sample'])/2)
		print(num_mislabel_per_ctl)
		for i in range(5, 10):
				idxs = np.where((self.train_labels == i) == True)[0]
				np.random.shuffle(idxs)
				for j in range(num_mislabel_per_ctl):
						self.train_labels[idxs[j]] = np.random.randint(0, 5)


	def asymmetric_noise(self):
		for i in range(self.num_classes):
			indices = np.where(self.train_labels == i)[0]
			np.random.shuffle(indices)
			for j, idx in enumerate(indices):
				if j < self.cfg_trainer['percent'] * len(indices):
					# truck -> automobile
					if i == 9:
						self.train_labels[idx] = 1
					# automobile -> truck
					elif i == 1:
						self.train_labels[idx] = 9
					# cat -> dog
					elif i == 3:
						self.train_labels[idx] = 5
					# dog -> cat
					elif i == 5:
						self.train_labels[idx] = 3
					# deer -> horse
					elif i == 4:
						self.train_labels[idx] = 7
					# horse -> deer
					elif i == 7:
						self.train_labels[idx] = 4
	def __len__(self):
		return len(self.train_data)


	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)


		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target, index, target_gt
		