import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import os
import json
from numpy.testing import assert_array_almost_equal

def fix_seed(seed=888):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

def get_cifar100(root, cfg_trainer, train=True,
				transform_train=None, transform_val=None,
				download=False, noise_file = '', teacher_idx=None, seed=888):
	base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)

	if cfg_trainer['asym']:
		if train:
			fix_seed(seed)
			train_idxs, val_idxs = train_val_split(base_dataset.targets, seed)
			train_dataset = CIFAR100_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, seed=seed)
			val_dataset = CIFAR100_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
			train_dataset.asymmetric_noise()
			val_dataset.asymmetric_noise()
				  
			if teacher_idx is not None:
				print(len(teacher_idx))
				train_dataset.truncate(teacher_idx)
			print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
		else:
			fix_seed(seed)
			train_dataset = []
			val_dataset = CIFAR100_val(root, cfg_trainer, None, train=train, transform=transform_val)
			print(f"Test: {len(val_dataset)}")

		return train_dataset, val_dataset
	elif cfg_trainer['control']:
		if train:
			fix_seed(seed)
			train_idxs, val_idxs = train_val_split_ctl(base_dataset.targets, cfg_trainer, seed)
			train_dataset = CIFAR100_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, seed=seed)
			val_dataset = CIFAR100_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
			train_dataset.control_noise()
			val_dataset.control_noise()
				  
			if teacher_idx is not None:
				print(len(teacher_idx))
				train_dataset.truncate(teacher_idx)
			print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
		else:
			fix_seed(seed)
			train_dataset = []
			val_dataset = CIFAR100_val(root, cfg_trainer, None, train=train, transform=transform_val)
			print(f"Test: {len(val_dataset)}")

		return train_dataset, val_dataset


def train_val_split(base_dataset: torchvision.datasets.CIFAR100, seed):
	fix_seed(seed)
	num_classes = 100
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
	num_classes = 100
	base_dataset = np.array(base_dataset)
	ctl_cls_lower_bound = 50
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


class CIFAR100_train(torchvision.datasets.CIFAR100):
	def __init__(self, root, cfg_trainer, indexs, train=True,
				 transform=None, target_transform=None,
				 download=False, seed=888):
		super(CIFAR100_train, self).__init__(root, train=train,
											transform=transform, target_transform=target_transform,
											download=download)
		fix_seed(seed)
		self.num_classes = 100
		self.cfg_trainer = cfg_trainer
		self.train_data = self.data[indexs]#self.train_data[indexs]
		self.train_labels = np.array(self.targets)[indexs]#np.array(self.train_labels)[indexs]
		self.indexs = indexs
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()
		self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
		self.noise_indx = []
		#self.all_refs_encoded = torch.zeros(self.num_classes,self.num_ref,1024, dtype=np.float32)
		self.seed = seed
		self.count = 0

	def symmetric_noise(self):
		self.train_labels_gt = self.train_labels.copy()
		indices = np.random.permutation(len(self.train_data))
		for i, idx in enumerate(indices):
			if i < self.cfg_trainer['percent'] * len(self.train_data):
				self.noise_indx.append(idx)
				self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()

	def control_noise(self):
		self.whole_train_labels_gt = self.train_labels.copy()
		control_lower_idx = 50
		num_ctl_cls = 50
		num_mislabel_per_ctl = int((0.9*self.cfg_trainer['ctl_cls_sample'] - 0.9*self.cfg_trainer['non_ctl_cls_sample'])/2)
		print(num_mislabel_per_ctl)
		for i in range(50, 100):
				idxs = np.where((self.train_labels == i) == True)[0]
				np.random.shuffle(idxs)
				for j in range(num_mislabel_per_ctl):
						self.train_labels[idxs[j]] = np.random.randint(0, 50)
		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()
		self.train_labels_gt = self.whole_train_labels_gt.copy()


	def multiclass_noisify(self, y, P, random_state=0):
		""" Flip classes according to transition probability matrix T.
		It expects a number between 0 and the number of classes - 1.
		"""

		assert P.shape[0] == P.shape[1]
		assert np.max(y) < P.shape[0]

		# row stochastic matrix
		assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
		assert (P >= 0.0).all()

		m = y.shape[0]
		new_y = y.copy()
		flipper = np.random.RandomState(random_state)

		for idx in np.arange(m):
			i = y[idx]
			# draw a vector with only an 1
			flipped = flipper.multinomial(1, P[i, :], 1)[0]
			new_y[idx] = np.where(flipped == 1)[0]

		return new_y

#     def build_for_cifar100(self, size, noise):
#         """ random flip between two random classes.
#         """
#         assert(noise >= 0.) and (noise <= 1.)

#         P = np.eye(size)
#         cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
#         P[cls1, cls2] = noise
#         P[cls2, cls1] = noise
#         P[cls1, cls1] = 1.0 - noise
#         P[cls2, cls2] = 1.0 - noise

#         assert_array_almost_equal(P.sum(axis=1), 1, 1)
#         return P
	def build_for_cifar100(self, size, noise):
		""" The noise matrix flips to the "next" class with probability 'noise'.
		"""

		assert(noise >= 0.) and (noise <= 1.)

		P = (1. - noise) * np.eye(size)
		for i in np.arange(size - 1):
			P[i, i + 1] = noise

		# adjust last row
		P[size - 1, 0] = noise

		assert_array_almost_equal(P.sum(axis=1), 1, 1)
		return P


	def asymmetric_noise(self):
		fix_seed(self.seed)
		self.whole_train_labels_gt = self.train_labels.copy()
		for i in range(self.num_classes):
			indices = np.where(self.train_labels == i)[0]
			np.random.shuffle(indices)
			for j, idx in enumerate(indices):
				if j < self.cfg_trainer['percent'] * len(indices):
					self.noise_indx.append(idx)
					# beaver (4) -> otter (55)
					# aquarium fish (1) -> flatfish (32)
					# poppies (62)-> roses (70)
					# bottles (9) -> cans (16)
					# apples (0) -> pears (57)
					# chair (20) -> couch (25)
					# bee (6) -> beetle (7)
					# lion (43)-> tiger (88)
					# crab (26) -> spider (79)
					# rabbit (65) -> squirrel (80)
					# maple (47) -> oak (52)
					# bicycle (8)-> motorcycle (48)
					if i == 4:
						self.train_labels[idx] = 55
					elif i == 1:
						self.train_labels[idx] = 32
					elif i == 62:
						self.train_labels[idx] = 70
					elif i == 9:
						self.train_labels[idx] = 16
					elif i == 0:
						self.train_labels[idx] = 57
					elif i == 20:
						self.train_labels[idx] = 25
					elif i == 6:
						self.train_labels[idx] = 7
					elif i == 43:
						self.train_labels[idx] = 88
					elif i == 26:
						self.train_labels[idx] = 79
					elif i == 65:
						self.train_labels[idx] = 80
					elif i == 47:
						self.train_labels[idx] = 52
					elif i == 8:
						self.train_labels[idx] = 48

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
		for i in range(0, 100):
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
		img, target, target_gt = self.train_data[index], self.train_labels[index],  self.train_labels_gt[index]


		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)


		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target, index, target_gt

	def __len__(self):
		return len(self.train_data)


class CIFAR100_val(torchvision.datasets.CIFAR100):

	def __init__(self, root, cfg_trainer, indexs, train=True,
				 transform=None, target_transform=None,
				 download=False):
		super(CIFAR100_val, self).__init__(root, train=train,
										  transform=transform, target_transform=target_transform,
										  download=download)

		# self.train_data = self.data[indexs]
		# self.train_labels = np.array(self.targets)[indexs]
		self.num_classes = 100
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
		control_lower_idx = 50
		num_ctl_cls = 50
		num_mislabel_per_ctl = int((0.1*self.cfg_trainer['ctl_cls_sample'] - 0.1*self.cfg_trainer['non_ctl_cls_sample'])/2)
		print(num_mislabel_per_ctl)
		for i in range(50, 100):
				idxs = np.where((self.train_labels == i) == True)[0]
				np.random.shuffle(idxs)
				for j in range(num_mislabel_per_ctl):
						self.train_labels[idxs[j]] = np.random.randint(0, 50)
				
	def multiclass_noisify(self, y, P, random_state=0):
		""" Flip classes according to transition probability matrix T.
		It expects a number between 0 and the number of classes - 1.
		"""

		assert P.shape[0] == P.shape[1]
		assert np.max(y) < P.shape[0]

		# row stochastic matrix
		assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
		assert (P >= 0.0).all()

		m = y.shape[0]
		new_y = y.copy()
		flipper = np.random.RandomState(random_state)

		for idx in np.arange(m):
			i = y[idx]
			# draw a vector with only an 1
			flipped = flipper.multinomial(1, P[i, :], 1)[0]
			new_y[idx] = np.where(flipped == 1)[0]

		return new_y

#     def build_for_cifar100(self, size, noise):
#         """ random flip between two random classes.
#         """
#         assert(noise >= 0.) and (noise <= 1.)

#         P = np.eye(size)
#         cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
#         P[cls1, cls2] = noise
#         P[cls2, cls1] = noise
#         P[cls1, cls1] = 1.0 - noise
#         P[cls2, cls2] = 1.0 - noise

#         assert_array_almost_equal(P.sum(axis=1), 1, 1)
#         return P
	def build_for_cifar100(self, size, noise):
		""" The noise matrix flips to the "next" class with probability 'noise'.
		"""

		assert(noise >= 0.) and (noise <= 1.)

		P = (1. - noise) * np.eye(size)
		for i in np.arange(size - 1):
			P[i, i + 1] = noise

		# adjust last row
		P[size - 1, 0] = noise

		assert_array_almost_equal(P.sum(axis=1), 1, 1)
		return P


	def asymmetric_noise(self):
		self.whole_train_labels_gt = self.train_labels.copy()
		for i in range(self.num_classes):
			indices = np.where(self.train_labels == i)[0]
			np.random.shuffle(indices)
			for j, idx in enumerate(indices):
				if j < self.cfg_trainer['percent'] * len(indices):
					# beaver (4) -> otter (55)
					# aquarium fish (1) -> flatfish (32)
					# poppies (62)-> roses (70)
					# bottles (9) -> cans (16)
					# apples (0) -> pears (57)
					# chair (20) -> couch (25)
					# bee (6) -> beetle (7)
					# lion (43)-> tiger (88)
					# crab (26) -> spider (79)
					# rabbit (65) -> squirrel (80)
					# maple (47) -> oak (52)
					# bicycle (8)-> motorcycle (48)
					if i == 4:
						self.train_labels[idx] = 55
					elif i == 1:
						self.train_labels[idx] = 32
					elif i == 62:
						self.train_labels[idx] = 70
					elif i == 9:
						self.train_labels[idx] = 16
					elif i == 0:
						self.train_labels[idx] = 57
					elif i == 20:
						self.train_labels[idx] = 25
					elif i == 6:
						self.train_labels[idx] = 7
					elif i == 43:
						self.train_labels[idx] = 88
					elif i == 26:
						self.train_labels[idx] = 79
					elif i == 65:
						self.train_labels[idx] = 80
					elif i == 47:
						self.train_labels[idx] = 52
					elif i == 8:
						self.train_labels[idx] = 48

		self.whole_train_data = self.train_data.copy()
		self.whole_train_labels = self.train_labels.copy()
		self.train_labels_gt = self.whole_train_labels_gt.copy()

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


	
