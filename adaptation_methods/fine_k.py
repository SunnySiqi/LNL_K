import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import pairwise_distances
from torch.autograd import grad
from sklearn.mixture import GaussianMixture as GMM
from tqdm import tqdm
import torch.nn.functional as F
from LitCNN import *


def get_singular_vector(singular_vector_dict, features, labels):
	'''
	To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
	features: hidden feature vectors of data (numpy)
	labels: correspoding label list
	'''
	
	with tqdm(total=len(np.unique(labels))) as pbar:
		for index in np.unique(labels):
			_, _, v = np.linalg.svd(features[labels==index])
			singular_vector_dict[index] = v[0]
			pbar.update(1)

	return singular_vector_dict

def get_score_w_noise_source(ctl_classid, singular_vector_dict, features, labels):
	'''
	helper function of the FINE+K method: class score shows the alignment between the class singular vector and itself, 
	final score is the difference between the label-class score and noise-source-class score.
	'''
	scores = []
	for i, feat in enumerate(features):
		label_i = labels[i]
		source_max_score = np.abs(np.inner(singular_vector_dict[ctl_classid], feat/np.linalg.norm(feat)))
		class_score = np.abs(np.inner(singular_vector_dict[label_i], feat/np.linalg.norm(feat)))
		scores.append(class_score - source_max_score)
	return np.array(scores)

def get_score(singular_vector_dict, features, labels):
	'''
	helper function of the FINE method: class score shows the alignment between the class singular vector and itself. 
	'''
	scores = []
	for i, feat in enumerate(features):
		label_i = labels[i]
		class_score = np.abs(np.inner(singular_vector_dict[label_i], feat/np.linalg.norm(feat)))
		scores.append(class_score)
	return np.array(scores)

def fit_mixture(scores, labels, p_threshold=0.5):
	'''
	Assume the distribution of scores: bimodal gaussian mixture model
	return clean labels that belongs to the clean cluster by fitting the score distribution to GMM
	'''

	clean_labels = []
	indexes = np.array(range(len(scores)))
	for cls in np.unique(labels):
		cls_index = indexes[labels==cls]
		feats = scores[labels==cls]
		feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
		gmm = GMM(n_components=2, covariance_type='full', tol=1e-6, max_iter=100)		
		gmm.fit(feats_)
		prob = gmm.predict_proba(feats_)
		prob = prob[:,gmm.means_.argmax()]
		clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold] 		
	return np.array(clean_labels)


class fine_LitCNN(LitCNN):
	def __init__(self, args, num_classes=1, train_data_module=None):
		super().__init__(self, args, num_classes=1, train_data_module=None)
		self.prev_features = None
		self.prev_labels = None
		self.vector_dict = {}
		self.ctl_targetid = 0
		self.all_features = None
		self.all_targets = None
		self.all_indexes = None


	def fine(self):
		ctl_ids = np.where((self.all_targets == self.ctl_targetid) == True)[0]
		ssets = list(self.all_indexes[ctl_ids])
		if self.prev_features is not None and self.prev_labels is not None:
			self.vector_dict = get_singular_vector(self.vector_dict, self.prev_features, self.prev_labels)
		else:
			self.vector_dict = get_singular_vector(self.vector_dict, self.all_features, self.all_targets)
		self.prev_features = self.all_features
		self.prev_labels = self.all_targets
		scores = get_score(self.vector_dict, features = self.all_features, labels = self.all_targets)
		clean_labels = fit_mixture(scores, self.all_targets, p_threshold=0.5)
		final_clean_labels = list(set(ssets)|set(self.all_indexes[clean_labels]))
		return final_clean_labels

	def fine_k(self):
		ctl_ids = np.where((self.all_targets == self.ctl_targetid) == True)[0]
		ssets = list(self.all_indexes[ctl_ids])
		if self.prev_features is not None and self.prev_labels is not None:
			self.vector_dict = get_singular_vector(self.vector_dict, self.prev_features, self.prev_labels)
		else:
			self.vector_dict = get_singular_vector(self.vector_dict, self.all_features, self.all_targets)
		self.prev_features = self.all_features
		self.prev_labels = self.all_targets
		scores = get_score_w_noise_source(self.ctl_targetid, self.vector_dict, features = self.all_features, labels = self.all_targets)
		clean_labels = fit_mixture(scores, self.all_targets, p_threshold=0.5)
		final_clean_labels = list(set(ssets)|set(self.all_indexes[clean_labels]))
		return final_clean_labels

	def training_epoch_end(self, training_step_outputs):
		all_preds = torch.cat([training_step_outputs[i]['preds'] for i in range(len(training_step_outputs))])
		all_targets = torch.cat([training_step_outputs[i]['targets']for i in range(len(training_step_outputs))])
		all_targets = all_targets.reshape((1,-1))

		all_preds = all_preds.reshape(-1, all_preds.shape[-1])
		all_preds = all_preds.detach().cpu().numpy()
		all_targets = all_targets.detach().cpu().numpy()
		all_targets = np.squeeze(all_targets)

		top1, top5, top10 = avg_accuracy(all_preds, all_targets)
		self.log("train_top1acc", top1)
		self.log("train_top5acc", top5)
		self.log("train_top10acc", top10)

        # Update dataloader with selected clean samples
		self.train_data_module.train_dataset.switch_data()
		self.all_targets, self.all_features, self.all_indexes = self.predict_to_select()
		
		if self.args.use_fine and self.trainer.current_epoch >= self.args.start_epoch:
			final_clean_samples = self.fine()
		elif self.args.use_fine_w and self.trainer.current_epoch >= self.args.start_epoch:
			final_clean_samples = self.fine_k()

		self.train_data_module.train_dataset.adjust_base_indx(final_clean_samples)
		print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
		print("change train loader")





