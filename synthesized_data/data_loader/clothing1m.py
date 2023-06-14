import sys
import os
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random

def fix_seed(seed=777):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

def get_clothing1m(root, cfg_trainer, num_samples=0, train=True,
				transform_train=None, transform_val=None, teacher_idx=None, easy_idx=None, hard_idx=None, seed=8888):

	if train:
		fix_seed(seed)
		train_dataset = Clothing1M_Dataset(root, cfg_trainer, num_samples=num_samples, train=train, transform=transform_train, seed=seed)
		val_dataset = Clothing1M_Dataset(root, cfg_trainer, val=train, transform=transform_val)
		print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")

	else:
		fix_seed(seed)
		train_dataset = []
		val_dataset = Clothing1M_Dataset(root, cfg_trainer, test= (not train), transform=transform_val)
		print(f"Test: {len(val_dataset)}")
		
	if easy_idx is not None and hard_idx is not None and train:
		train_dataset.mixup_and_add(hard_idx, teacher_idx, easy_idx)
	elif teacher_idx is not None and train:
		print (len(teacher_idx))
		train_dataset.truncate(teacher_idx)

	return train_dataset, val_dataset

class Clothing1M_Dataset(torch.utils.data.Dataset):

	def __init__(self, root, cfg_trainer, num_samples=0, train=False, val=False, test=False, transform=None, num_class=14, seed=8888):
		
		fix_seed(seed)
		self.cfg_trainer = cfg_trainer
		self.root = root
		self.transform = transform
		self.train_labels = {}
		self.test_labels = {}
		self.val_labels = {}  

		self.train  = train
		self.val = val
		self.test = test

		with open('%s/annotations/noisy_label_kv.txt'%self.root,'r') as f:
			lines = f.read().splitlines()
			for l in lines:
				entry = l.split()           
				img_path = '%s/'%self.root+entry[0][7:]
				self.train_labels[img_path] = int(entry[1])                         
		with open('%s/annotations/clean_label_kv.txt'%self.root,'r') as f:
			lines = f.read().splitlines()
			for l in lines:
				entry = l.split()           
				img_path = '%s/'%self.root+entry[0][7:]
				self.test_labels[img_path] = int(entry[1])  

		if train:          
			train_imgs=[]
			with open('%s/annotations/noisy_train_key_list.txt'%self.root,'r') as f:
				lines = f.read().splitlines()
				for i , l in enumerate(lines):
					img_path = '%s/'%self.root+l[7:]
					train_imgs.append((i,img_path)) 
			self.num_raw_example = len(train_imgs)                              
			random.shuffle(train_imgs)
			class_num = torch.zeros(num_class)
			self.train_imgs = []
			self.train_labels_ = []
			for id_raw, impath in train_imgs:
				label = self.train_labels[impath]
				if class_num[label] < (num_samples/14) and len(self.train_imgs)<num_samples:
					self.train_imgs.append((id_raw,impath))
					self.train_labels_.append(int(label))
					class_num[label]+=1
#                 else:
#                     print (label, class_num[label], (num_samples/14))
			random.shuffle(self.train_imgs)
			self.train_imgs = np.array(self.train_imgs)
			self.train_labels_ = np.array(self.train_labels_)
			self.train_label_dict = {}
			uniq_labels = np.unique(self.train_labels_)
			for uniq_label in uniq_labels:
				self.train_label_dict[uniq_label] = np.where(self.train_labels_ == uniq_label)[0]



		elif test:
			self.test_imgs = []
			with open('%s/annotations/clean_test_key_list.txt'%self.root,'r') as f:
				lines = f.read().splitlines()
				for l in lines:
					img_path = '%s/'%self.root+l[7:]
					self.test_imgs.append(img_path)            
		elif val:
			self.val_imgs = []
			with open('%s/annotations/clean_val_key_list.txt'%self.root,'r') as f:
				lines = f.read().splitlines()
				for l in lines:
					img_path = '%s/'%self.root+l[7:]
					self.val_imgs.append(img_path)

	def __getitem__(self, index):
		if self.train:
			if len(self.train_imgs[index]) == 4:
				id_raw0, img_path0, id_raw1, img_path1 = self.train_imgs[index]
				target =  self.train_labels[img_path0]
				image0 = self.transform(Image.open(img_path0).convert('RGB'))
				image1 = self.transform(Image.open(img_path1).convert('RGB'))
				lam = np.random.beta(1, 0.1)
				new_image = lam * image0 + (1 - lam) * image1
				return new_image, target, id_raw0, target
			else:
				id_raw, img_path = self.train_imgs[index]
				target = self.train_labels[img_path] 
				img0 = self.transform(Image.open(img_path).convert('RGB')) 
				return img0, target, id_raw, target   
		elif self.val:
			img_path = self.val_imgs[index]
			target = self.test_labels[img_path]   
		elif self.test:
			img_path = self.test_imgs[index]
			target = self.test_labels[img_path] 
		img = self.transform(Image.open(img_path).convert('RGB'))
		return img, target, index, target


	def __len__(self):
		if self.test:
			return len(self.test_imgs)
		if self.val:
			return len(self.val_imgs)
		else:
			return len(self.train_imgs) 

	def flist_reader(self, flist):
		imlist = []
		with open(flist, 'r') as rf:
			for line in rf.readlines():
				row = line.split(" ")
				impath =  self.root + row[0]
				imlabel = float(row[1].replace('\n',''))
				imlist.append((impath, int(imlabel)))
		return imlist
	
	def truncate(self, teacher_idx):
		self.train_imgs = self.train_imgs[teacher_idx]
		self.train_labels_ = self.train_labels_[teacher_idx]


	def mixup_and_add(self, hard_idx, mid_idx, easy_idx):
		hard_label_dict = {}
		easy_label_dict = {}
		for unique_label in self.train_label_dict:
			hard_label_dict[unique_label] = list(set(hard_idx).intersection(set(self.train_label_dict[unique_label])))
			easy_label_dict[unique_label] = list(set(easy_idx).intersection(set(self.train_label_dict[unique_label])))
		# images = list(self.train_imgs)
		# labels = list(self.train_labels_)
		images = []
		labels = []
		for hard in hard_idx:
			label = self.train_labels_[hard]
			random_easy = np.random.choice(easy_label_dict[label], 1)[0]
			id0, img0 = self.train_imgs[hard]
			id1, img1 = self.train_imgs[random_easy]
			images.append((id0, img0, id1, img1))
			labels.append(int(label))
		for easy in easy_idx:
			label = self.train_labels_[easy]
			random_hard = np.random.choice(hard_label_dict[label], 1)[0]
			id0, img0 = self.train_imgs[easy]
			id1, img1 = self.train_imgs[random_hard]
			images.append((id0, img0, id1, img1))
			labels.append(int(label))
		images += list(self.train_imgs[mid_idx])
		labels += list(self.train_labels_[mid_idx])
		self.train_imgs = np.array(images)
		self.train_labels_ = np.array(labels)














