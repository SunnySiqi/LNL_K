import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import ConfusionMatrix
import torch.distributions as dd
from sklearn.metrics import pairwise_distances
import heapq
from torch.autograd import grad
from sklearn.mixture import GaussianMixture as GMM
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


def get_distribution(features):
	return dd.normal.Normal(torch.tensor(np.mean(features, axis=0)), torch.tensor(np.std(features, axis=0)+1e-9))

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
	scores = []
	for i, feat in enumerate(features):
		label_i = labels[i]
		source_max_score = np.abs(np.inner(singular_vector_dict[ctl_classid], feat/np.linalg.norm(feat)))
		class_score = np.abs(np.inner(singular_vector_dict[label_i], feat/np.linalg.norm(feat)))
		scores.append(class_score - source_max_score)
	return np.array(scores)

def get_score(singular_vector_dict, features, labels):
	scores = []
	for i, feat in enumerate(features):
		label_i = labels[i]
		class_score = np.abs(np.inner(singular_vector_dict[label_i], feat/np.linalg.norm(feat)))
		scores.append(class_score)
	return np.array(scores)

def fit_mixture(scores, labels, p_threshold=0.5):
	'''
	Assume the distribution of scores: bimodal gaussian mixture model
	
	return clean labels
	that belongs to the clean cluster by fitting the score distribution to GMM
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

class GrowingCluster(object):
	def __init__(self, startpoint):
		self.points = [startpoint]
		self.centroid = startpoint
	def add_points(self, newpoint):
		self.points.append(newpoint)
		self.centroid = np.mean(self.points, axis=0)

def sset_distribution_init(args, ctl_centroids, c_features):
	if args.naive_init:
		return get_distribution(c_features)
	ctl_centroids = ctl_centroids.reshape(1,-1)
	startpoint_idx = np.argmax(np.min(pairwise_distances(c_features, ctl_centroids), axis=1))
	c_cluster = GrowingCluster(c_features[startpoint_idx])
	targets = ctl_centroids + [c_cluster.centroid]
	c_label = len(targets)-1
	candidates = np.delete(c_features, startpoint_idx, axis=0)
	while len(candidates) >0:
		targets[-1] = c_cluster.centroid
		dist = pairwise_distances(candidates, targets)
		label = np.argmin(dist, axis=1)
		dist2c = dist[:,-1]
		new_id = np.argmin(dist2c)
		new_ids = np.where((label == c_label) == True)[0]
		if new_id not in new_ids:
			break    
		c_cluster.add_points(candidates[new_id])
		candidates = np.delete(candidates, new_id, axis=0)
	# distribution = get_distribution(np.array(c_cluster.points))
	# for f in c_cluster.points:
	#     print(f.shape)
	#     print(distribution.log_prob(torch.tensor(f)))
	return get_distribution(np.array(c_cluster.points))

def accuracy(output, target):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	topk = (1, 5, 10)
	maxk = max(topk)
	target_size = len(target)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1,).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / target_size))
	return res

def avg_accuracy(output, target):
	# output = output.detach().cpu().numpy()
	# target = target.detach().cpu().numpy()
	cls_list = list(set(target))
	num_cls = len(cls_list)
	acc_top1 = [0]*num_cls
	acc_top5 = [0]*num_cls
	acc_top10 = [0]*num_cls
	for i in range(num_cls):
		c = cls_list[i]
		c_ids = np.where((target == c) == True)[0]
		acc_top1[i], acc_top5[i], acc_top10[i] = accuracy(torch.Tensor(output[c_ids]), torch.Tensor(target[c_ids]))
	acc_top1 = torch.cat(acc_top1)
	acc_top5 = torch.cat(acc_top5)
	acc_top10 = torch.cat(acc_top10)
	print(acc_top1)
	print(acc_top5)
	print(acc_top10)
	return [acc_top1.mean(), acc_top5.mean(), acc_top10.mean()]

def avg_accuracy_with_idx(output, target, num_classes):
	acc_top1 = [0]*num_classes
	acc_top5 = [0]*num_classes
	acc_top10 = [0]*num_classes
	for i in range(num_classes):
		c_ids = np.where((target == i) == True)[0]
		acc_top1[i], acc_top5[i], acc_top10[i] = accuracy(torch.Tensor(output[c_ids]), torch.Tensor(target[c_ids]))
	acc_top1 = torch.cat(acc_top1)
	acc_top5 = torch.cat(acc_top5)
	acc_top10 = torch.cat(acc_top10)
	print(acc_top1)
	print(acc_top5)
	print(acc_top10)
	return [acc_top1.mean(), acc_top5.mean(), acc_top10.mean()]


#For Crust

def _heappush_max(heap, item):
	heap.append(item)
	heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
	"""Maxheap version of a heappop."""
	lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
	if heap:
		returnitem = heap[0]
		heap[0] = lastelt
		heapq._siftup_max(heap, 0)
		return returnitem
	return lastelt


def lazy_greedy_heap(F, V, B):
	curVal = 0
	sset = []
	vals = []

	order = []
	heapq._heapify_max(order)
	# [_heappush_max(order, (F.inc(sset, index), index)) for index in V]
	cnt = 0
	for index in V:
	  _heappush_max(order, (F.inc(sset, index), index))
	  cnt += 1

	n_iter = 0
	while order and len(sset) < B:
		n_iter += 1
		if F.curVal == len(F.D):
		  # all points covered
		  break

		el = _heappop_max(order)
		improv = F.inc(sset, el[1])

		# check for uniques elements
		if improv > 0: 
			if not order:
				curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
				sset.append(el[1])
				vals.append(curVal)
			else:
				top = _heappop_max(order)
				if improv >= top[0]:
					curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
					sset.append(el[1])
					vals.append(curVal)
				else:
					_heappush_max(order, (improv, el[1]))
				_heappush_max(order, top)

	return sset, vals

class FacilityLocation:
	def __init__(self, V, D=None, fnpy=None):
		if D is not None:
		  self.D = D
		else:
		  self.D = np.load(fnpy)

		self.D *= -1
		self.D -= self.D.min()
		self.V = V
		self.curVal = 0
		self.gains = []
		self.curr_max = np.zeros_like(self.D[0])

	def inc(self, sset, ndx):
		if len(sset + [ndx]) > 1:
			new_dists = np.stack([self.curr_max, self.D[ndx]], axis=0)
			return new_dists.max(axis=0).sum()
		else:
			return self.D[sset + [ndx]].sum()

	def add(self, sset, ndx, delta):
		self.curVal += delta
		self.gains += delta,
		self.curr_max = np.stack([self.curr_max, self.D[ndx]], axis=0).max(axis=0)
		return self.curVal

		cur_old = self.curVal
		if len(sset + [ndx]) > 1:
			self.curVal = self.D[:, sset + [ndx]].max(axis=1).sum()
		else:
			self.curVal = self.D[:, sset + [ndx]].sum()
		self.gains.extend([self.curVal - cur_old])
		return self.curVal
# Self-Filter Loss
class SelfFilterLoss(nn.Module):
	def __init__(self, num_classes):
		super(SelfFilterLoss, self).__init__()
		self.num_classes = num_classes

	def one_lossF(self, output, one_hot):
		log_prob = torch.nn.functional.log_softmax(output, dim=1)
		loss = - torch.sum(log_prob * one_hot) / output.size(0)
		return loss

	def forward(self, output, target, mode=None):
		loss_ce = F.cross_entropy(output, target)
		bs = output.size()[0]
		pseudo_label = torch.softmax(output, dim=1)
		if mode == 'warm_up':
			values, index = pseudo_label.topk(k=2, dim=1)
			latent_label = index[:, 1]
			latent_lam = torch.zeros(pseudo_label.size(0), 1).cuda().float()

			for i in range(values.size(0)):
				x = values[i]
				latent_lam[i] = max(0.2 - min(x) / max(x), 0.0)
			conf_penalty = 0.5 * (F.cross_entropy(output, latent_label, reduction='none') * latent_lam).mean()
		elif mode == 'train':
			latent_labels = torch.zeros((bs, self.num_classes)).cuda()
			for i in range(bs):
				confident = pseudo_label[i]
				max_ = confident[target[i]]
				confident = 0.2 - confident / max_
				mask = (confident >= 0.0)
				latent_labels[i] = confident * mask
			conf_penalty = 0.1 * self.one_lossF(output, latent_labels) / (self.num_classes - 1)
		else:
			raise ValueError('')

		loss = loss_ce + conf_penalty
		return loss

#define the LightningModule
class LitCNN(pl.LightningModule):
	def __init__(self, args, num_classes=1, train_data_module=None):
		super().__init__()
		self.num_classes = num_classes
		self.args = args
		self.best_acc = 0.0
		self.train_data_module = train_data_module
		self.prev_features = None
		self.prev_labels = None
		self.vector_dict = {}
		net = models.efficientnet_b0(weights='DEFAULT')
		layers = list(net.children())
		layers[0][0][0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
		if self.args.net == 'full_effb0':
			layers = list(net.children())[:-1]
			self.net = nn.Sequential(*layers)
			self.linear = nn.Linear(1280, num_classes)
		elif self.args.net == 'effb0':
			layers = list(layers[0][:6]) + list(list(layers[0][6][0].children())[0][:2])
			layers.append(nn.AdaptiveAvgPool2d(output_size=1))
			self.net = nn.Sequential(*layers)
			self.linear = nn.Linear(672, num_classes)
		elif self.args.net == 'res50':
			net = models.resnet50(pretrained=True)
			net.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
			layers = list(net.children())[:-1]
			self.net = nn.Sequential(*layers)
			self.linear = nn.Linear(2048, num_classes)
		else:
			net = models.resnet18(pretrained=True)
			net.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
			layers = list(net.children())[:-1]
			self.net = nn.Sequential(*layers)
			self.linear = nn.Linear(512, num_classes)
		if self.args.use_self_filter or self.args.use_self_filter_w:
			self.memory_bank = []
			self.self_filter_loss = SelfFilterLoss(self.num_classes)

	# def __init__(self):
	# 	super().__init__()
	# 	self.num_classes = 100
	# 	self.best_acc = 0.0
	# 	self.prev_features = None
	# 	self.prev_labels = None
	# 	self.vector_dict = {}
	# 	net = models.efficientnet_b0(weights='DEFAULT')
	# 	layers = list(net.children())
	# 	layers[0][0][0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
	# 	layers = list(net.children())[:-1]
	# 	self.net = nn.Sequential(*layers)
	# 	self.linear = nn.Linear(1280, self.num_classes)
		

	def forward(self, x):
		#x = x.view(x.size(0), -1)
		z = self.net(x)
		feature = z.view(z.size(0), -1)
		#x_hat = self.activate(self.linear(feature))
		x_hat = self.linear(feature)
		return feature, x_hat

	def predict_to_select(self):
		all_features = []
		all_targets = []
		all_indexes = []
		data_loader = self.train_data_module.predict_dataloader()
		# print("DEVICE!!", self.device)
		for i, (imgs, target, index) in enumerate(data_loader):
			imgs = imgs.to(self.device)
			z = self.net(imgs)
			feature = z.view(z.size(0), -1)
			all_features.append(feature.detach().cpu().numpy())
			all_targets.append(target.detach().cpu().numpy())
			all_indexes.append(index)
		all_targets = np.hstack(all_targets)
		all_indexes = np.hstack(all_indexes)
		all_features = np.concatenate(all_features, axis=0)
		return all_targets, all_features, all_indexes

	def estimate_grads(self):
		all_grads = []
		all_targets = []
		all_preds = []
		all_indexes = []
		data_loader = self.train_data_module.predict_dataloader()
		for i, (imgs, target, index) in enumerate(data_loader):
			imgs = imgs.to(self.device)
			all_targets.append(target)
			target = target.to(self.device)
			# compute output
			feat, y_hat = self(imgs)
			_, pred = torch.max(y_hat, 1)
			loss = nn.CrossEntropyLoss(reduction='none')(y_hat, target).mean()
			est_grad = grad(loss, feat)
			all_grads.append(est_grad[0].detach().cpu().numpy())
			all_preds.append(pred.detach().cpu().numpy())
			all_indexes.append(index)
		all_grads = np.vstack(all_grads)
		all_targets = np.hstack(all_targets)
		all_indexes = np.hstack(all_indexes)
		all_preds = np.hstack(all_preds)
		return all_grads, all_targets, all_indexes

	def estimate_grads_w_noise_knowledge(self, control_label):
	# switch to train mode
		all_grads_dict = {}
		all_grads_dict['self'] = []
		all_grads_dict['control'] = []
		all_targets = []
		all_indexes = []
		data_loader = self.train_data_module.predict_dataloader()
		for i, (imgs, target, index) in enumerate(data_loader):
			imgs = imgs.to(self.device)
			all_targets.append(target)
			all_indexes.append(index)
			target = target.to(self.device)
			# compute output
			feat, y_hat = self(imgs)
			loss = nn.CrossEntropyLoss(reduction='none')(y_hat, target).mean()
			est_grad = grad(loss, feat)
			all_grads_dict['self'].append(est_grad[0].detach().cpu().numpy())
			ctl_label = torch.tensor(np.repeat(control_label, len(imgs)))
			ctl_label = ctl_label.to(self.device)
			feat, y_hat = self(imgs)
			loss_ctl = nn.CrossEntropyLoss(reduction='none')(y_hat, ctl_label).mean()
			est_grad_ctl = grad(loss_ctl, feat)
			all_grads_dict['control'].append(est_grad_ctl[0].detach().cpu().numpy())

		for dict_key in all_grads_dict:
			all_grads_dict[dict_key] = np.vstack(all_grads_dict[dict_key])
		all_targets = np.hstack(all_targets)
		all_indexes = np.hstack(all_indexes)
		return all_grads_dict, all_targets, all_indexes

	# For Self-Filter
	def get_memorybank(self, selection):
		k = int(self.args.self_filter_k)

		if selection:
			memory_bank_last = self.memory_bank[-1]
			fluctuation_ = [0] * 620000

		result = [0] * 620000
		data_loader = self.train_data_module.predict_dataloader()
		for i, (imgs, target, index) in enumerate(data_loader):
			imgs = imgs.to(self.device)
			feat, y_hat = self(imgs)
			max_probs, pred = y_hat.max(1)
			max_probs = max_probs.detach().cpu().numpy()
			pred = pred.detach().cpu().numpy()
			for b in range(index.size(0)):
				sample_index = index[b]
				if pred[b] == target[b]:
					result[sample_index] = max_probs[b]
				else:
					result[sample_index] = 0
				if selection:
				# calculate the fluctuation
					if memory_bank_last[sample_index] > result[sample_index] and result[sample_index] == 0:
					# the prediction of this sample changes from correct to wrong, thus it is fluctuation. we consider it as the clean with 0%
						fluctuation_[sample_index] = 0
					else:
						fluctuation_[sample_index] = 1

		# In practice, the fluctuation of predictions is easily influenced by SGD optimizer especially in extreme noise ratio,
		# Here, we design a smooth way by adding the confidence of prediction to fluctuation.
		# For that, the criterion will select the sample with high confidence even if there is a fluctuation
		if selection:
			for i in range(k - 1):
				self.memory_bank[i] = self.memory_bank[i + 1]
			self.memory_bank[-1] = result

			confidence_smooth = np.array([0.]* len(self.memory_bank[0]))
			for i in range(k):
				confidence_smooth += self.memory_bank[i]
			prob = (np.array(confidence_smooth) + np.array(fluctuation_)) / (k + 1)  # adding confidence make fluctuation more smooth
			pred = (prob > 0.5)
			return prob, pred

		else:
			if len(self.memory_bank) < k:
				self.memory_bank.append(result)
			else:
				for i in range(k - 1):
					self.memory_bank[i] = self.memory_bank[i + 1]
				self.memory_bank[-1] = result
			return

	def get_memorybank_w(self, selection):
		k = int(self.args.self_filter_k)

		if selection:
			memory_bank_last = self.memory_bank[-1]
			fluctuation_ = [0] * 620000

		clean_sset_id = []
		result = [0] * 620000
		data_loader = self.train_data_module.predict_dataloader()
		for i, (imgs, target, index) in enumerate(data_loader):
			imgs = imgs.to(self.device)
			target = target.to(self.device)
			feat, y_hat = self(imgs)
			max_probs, pred = y_hat.max(1)
			max_probs = max_probs.detach().cpu().numpy()
			pred = pred.detach().cpu().numpy()
			for b in range(index.size(0)):
				sample_index = index[b]
				if target[b] == 0:
					clean_sset_id.append(sample_index)
				elif pred[b] == target[b]:
					result[sample_index] = max_probs[b]
				elif pred[b] == 0:
					result[sample_index] = -max_probs[b]
				else:
					result[sample_index] = 0
				if selection:
				# calculate the fluctuation
					#if memory_bank_last[sample_index] < 0 and result[sample_index] >= 0 or memory_bank_last[sample_index] >= 0 and result[sample_index] < 0:
					# the prediction of this sample changes from correct to wrong, thus it is fluctuation. we consider it as the clean with 0%
					if memory_bank_last[sample_index] > 0 and result[sample_index] < 0:
						fluctuation_[sample_index] = 0
					else:
						fluctuation_[sample_index] = 1

		# In practice, the fluctuation of predictions is easily influenced by SGD optimizer especially in extreme noise ratio,
		# Here, we design a smooth way by adding the confidence of prediction to fluctuation.
		# For that, the criterion will select the sample with high confidence even if there is a fluctuation
		if selection:
			for i in range(k - 1):
				self.memory_bank[i] = self.memory_bank[i + 1]
			self.memory_bank[-1] = result

			confidence_smooth = np.array([0.]* len(self.memory_bank[0]))
			for i in range(k):
				confidence_smooth += self.memory_bank[i]
			prob = (np.array(confidence_smooth) + np.array(fluctuation_)) / (k + 1)  # adding confidence make fluctuation more smooth
			#pred = (prob > 0.5)
			prob[clean_sset_id] = 1
			pred = (prob > 0.5)
			return prob, pred

		else:
			if len(self.memory_bank) < k:
				self.memory_bank.append(result)
			else:
				for i in range(k - 1):
					self.memory_bank[i] = self.memory_bank[i + 1]
				self.memory_bank[-1] = result
			return

	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		# it is independent of forward
		x, y, _ = batch
		feature, y_hat = self(x)
		if self.args.use_self_filter or self.args.use_self_filter_w:
			if self.trainer.current_epoch < self.args.start_epoch:
				loss = self.self_filter_loss(y_hat, y, 'warm_up')
			else:
				loss = self.self_filter_loss(y_hat, y, 'train')
		else:
			loss = nn.CrossEntropyLoss(reduction='none')(y_hat, y)
		loss = loss.mean()
		# top1, top5, top10 = accuracy(x_hat, y)
		# Logging to TensorBoard by default
		self.log("train_loss", loss)
		# self.log("train_top1acc", top1)
		# self.log("train_top5acc", top5)
		# self.log("train_top10acc", top10)
		# return {'loss':loss, 'top1-acc':top1, 'top5-acc':top5, 'top10-acc':top10}
		#print("FINISHED!", str(batch_idx) + "_" + str(self.trainer.global_rank))
		return {'loss':loss, 'preds': y_hat, 'targets': y, 'features': feature}

	# def training_step_end(self, batch_parts):
	# 	print("TRAINER RANK: ", self.trainer.global_rank)
	# 	 # predictions from each GPU
	# 	predictions = batch_parts["preds"]
	# 	print("PREDICTION SHAPE", predictions.shape)
	# 	# losses from each GPU
	# 	targets = batch_parts['targets']
	# 	targets = torch.unsqueeze(targets, dim=1)
	# 	print("TARGETS SHAPE", targets.shape)
	# 	# features from each GPU
	# 	features = batch_parts['features']
	# 	print("FEATURES SHAPE", features.shape)

	# 	all_preds = torch.cat([predictions[i] for i in range(len(predictions))])
	# 	all_targets = torch.cat([targets[i] for i in range(len(targets))])
	# 	all_features = torch.cat([features[i] for i in range(len(features))])

	# 	print("PREDICTION SHAPE", all_preds.shape)
	# 	print("TARGETS SHAPE", all_targets.shape)
	# 	print("FEATURES SHAPE", all_features.shape)

	# 	return {'loss': batch_parts['loss'], 'all_preds': all_preds, 'all_targets': all_targets, 'all_features': all_features}


	def training_epoch_end(self, training_step_outputs):
		all_preds = torch.cat([training_step_outputs[i]['preds'] for i in range(len(training_step_outputs))])
		all_targets = torch.cat([training_step_outputs[i]['targets']for i in range(len(training_step_outputs))])
		# all_targets = pl.utilities.distributed.all_gather_ddp_if_available(all_targets)
		all_targets = all_targets.reshape((1,-1))
		# print("EPOCH END TARGETS SHAPE", all_targets.shape)
		# all_preds = pl.utilities.distributed.all_gather_ddp_if_available(all_preds)
		all_preds = all_preds.reshape(-1, all_preds.shape[-1])
		all_preds = all_preds.detach().cpu().numpy()
		all_targets = all_targets.detach().cpu().numpy()
		all_targets = np.squeeze(all_targets)
		# print("CHECK STATUS all_targets: ", str(np.shape(all_targets)) + '_' + str(self.trainer.global_rank))
		top1, top5, top10 = avg_accuracy(all_preds, all_targets)
		self.log("train_top1acc", top1)
		self.log("train_top5acc", top5)
		self.log("train_top10acc", top10)

		if self.trainer.current_epoch < self.args.start_epoch: 
			if self.args.use_self_filter:
				self.get_memorybank(False)
			elif self.args.use_self_filter_w:
				self.get_memorybank_w(False)

		if self.args.use_newgmm and self.best_acc > self.args.newgmm_startacc:
			# all_targets = pl.utilities.distributed.all_gather_ddp_if_available(all_targets)
			# all_targets = all_targets.reshape((1,-1))
			# all_features = torch.cat([training_step_outputs[i]['features'] for i in range(len(training_step_outputs))])
			# all_features = pl.utilities.distributed.all_gather_ddp_if_available(all_features)
			# all_features = all_features.reshape(-1, all_features.shape[-1])
			# #if self.trainer.global_rank == 0:
			# all_features = all_features.detach().cpu().numpy()
			# all_targets = all_targets.detach().cpu().numpy()
			# all_targets = np.squeeze(all_targets)
			self.train_data_module.train_dataset.switch_data()
			all_targets, all_features = self.predict_to_select()
			print("FEATURES", str(self.trainer.global_rank) + "_" + str(all_features.shape))
			print("TARGETS", str(self.trainer.global_rank) + "_" + str(all_targets.shape))

			ssets = []
			# ctl_targetid = self.train_data_module.train_dataset.treatment2index['control']
			ctl_targetid = 0
			coreset_cls_range = list(set(all_targets)-{ctl_targetid})
			ctl_distribution = get_distribution(all_features[np.where((all_targets == ctl_targetid) == True)[0]])
			ctl_centroid = np.mean(all_features[np.where((all_targets == ctl_targetid) == True)[0]], axis=0)
			for c in coreset_cls_range:
				sample_ids = np.where((all_targets == c) == True)[0]
				c_features = all_features[sample_ids]
				c_distribution = sset_distribution_init(self.args, ctl_centroid, c_features)
				ctl_log_prob = np.sum(ctl_distribution.log_prob(torch.tensor(c_features)).numpy(), axis=-1)
				ctl_score = ctl_log_prob
				c_log_prob = np.sum(c_distribution.log_prob(torch.tensor(c_features)).numpy(), axis=-1)
				prob_diff = c_log_prob - ctl_score
				threshold = np.quantile(prob_diff, 1-self.args.gmm_ratio)
				sset_ids = np.where((prob_diff > threshold) == True)[0]
				sset = sample_ids[np.array(sset_ids)]
				ssets += list(sset)
			ctl_ids = np.where((all_targets == ctl_targetid) == True)[0]
			ssets += list(ctl_ids)
			self.train_data_module.train_dataset.adjust_base_indx(ssets)
			print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
			print("change train loader")

		elif self.args.use_crust and self.trainer.current_epoch >= self.args.start_epoch:
			self.train_data_module.train_dataset.switch_data()      
			# FL part
			grads_all, labels, indexes = self.estimate_grads()
			# per-class clustering
			ssets = []
			# ctl_targetid = self.train_data_module.train_dataset.treatment2index['control']
			ctl_targetid = 0
			coreset_cls_range = list(set(all_targets)-{ctl_targetid})
			for c in coreset_cls_range:
				sample_ids = np.where((labels == c) == True)[0]
				grads = grads_all[sample_ids]
				dists = pairwise_distances(grads)
				V = range(len(grads))
				F = FacilityLocation(V, D=dists)
				B = int(self.args.fl_ratio * len(grads))
				sset, vals = lazy_greedy_heap(F, V, B)
				sset = indexes[sample_ids[np.array(sset)]]
				ssets += list(sset)
			ctl_ids = np.where((all_targets == ctl_targetid) == True)[0]
			ssets += list(indexes[ctl_ids])
			self.train_data_module.train_dataset.adjust_base_indx(ssets)
			print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
			print("change train loader")

		elif self.args.use_crust_w and self.trainer.current_epoch >= self.args.start_epoch:
			self.train_data_module.train_dataset.switch_data()      
			# FL part
			# ctl_targetid = self.train_data_module.train_dataset.treatment2index['control']
			ctl_targetid = 0
			grads_all_dict, labels, indexes = self.estimate_grads_w_noise_knowledge(ctl_targetid)
			# per-class clustering
			ssets = []
			coreset_cls_range = list(set(all_targets)-{ctl_targetid})
			ctl_ids = np.where((labels == ctl_targetid) == True)[0]
			ctl_grads = grads_all_dict['self'][ctl_ids]
			for c in coreset_cls_range:
				sample_ids = np.where((labels == c) == True)[0]
				grads_self = grads_all_dict['self'][sample_ids]
				sset = sample_ids
				grad_ctl = grads_all_dict['control'][sset]
				grads_all = np.concatenate((grad_ctl, ctl_grads), axis=0)
				sample_ids_all = np.concatenate((sset, ctl_ids), axis=0)
				dists = pairwise_distances(grads_all)
				V = range(len(grads_all))
				F = FacilityLocation(V, D=dists)
				B = int((1-self.args.fl_ratio) * len(grad_ctl) + len(ctl_ids))
				noisy_sset, vals = lazy_greedy_heap(F, V, B)
				noisy_sset = sample_ids_all[np.array(noisy_sset)]
				sset = indexes[np.array(list(set(sset).difference(set(noisy_sset))))]
				ssets += list(sset)
			ssets += list(indexes[ctl_ids])
			self.train_data_module.train_dataset.adjust_base_indx(ssets)
			print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
			print("change train loader")


		elif self.args.use_fine and self.trainer.current_epoch >= self.args.start_epoch:
			self.train_data_module.train_dataset.switch_data()
			all_targets, all_features, all_indexes = self.predict_to_select()
			# ctl_targetid = self.train_data_module.train_dataset.treatment2index['control']
			ctl_targetid = 0
			coreset_cls_range = list(set(all_targets)-{ctl_targetid})
			ctl_ids = np.where((all_targets == ctl_targetid) == True)[0]
			ssets = list(all_indexes[ctl_ids])
			if self.prev_features is not None and self.prev_labels is not None:
				self.vector_dict = get_singular_vector(self.vector_dict, self.prev_features, self.prev_labels)
			else:
				self.vector_dict = get_singular_vector(self.vector_dict, all_features, all_targets)
				self.prev_features = all_features
				self.prev_labels = all_targets
			scores = get_score(self.vector_dict, features = all_features, labels = all_targets)
			clean_labels = fit_mixture(scores, all_targets, p_threshold=0.5)
			final_clean_labels = list(set(ssets)|set(all_indexes[clean_labels]))
			self.train_data_module.train_dataset.adjust_base_indx(final_clean_labels)
			print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
			print("change train loader")

		if self.args.use_fine_w and self.trainer.current_epoch >= self.args.start_epoch:
			self.train_data_module.train_dataset.switch_data()
			all_targets, all_features, all_indexes = self.predict_to_select()
			# ctl_targetid = self.train_data_module.train_dataset.treatment2index['control']
			ctl_targetid = 0
			coreset_cls_range = list(set(all_targets)-{ctl_targetid})
			ctl_ids = np.where((all_targets == ctl_targetid) == True)[0]
			ssets = list(all_indexes[ctl_ids])
			if self.prev_features is not None and self.prev_labels is not None:
				self.vector_dict = get_singular_vector(self.vector_dict, self.prev_features, self.prev_labels)
			else:
				self.vector_dict = get_singular_vector(self.vector_dict, all_features, all_targets)
				self.prev_features = all_features
				self.prev_labels = all_targets
			scores = get_score_w_noise_source(ctl_targetid, self.vector_dict, features = all_features, labels = all_targets)
			clean_labels = fit_mixture(scores, all_targets, p_threshold=0.5)
			final_clean_labels = list(set(ssets)|set(all_indexes[clean_labels]))
			self.train_data_module.train_dataset.adjust_base_indx(final_clean_labels)
			print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
			print("change train loader")

		if self.args.use_self_filter and self.trainer.current_epoch >= self.args.start_epoch:
			self.train_data_module.train_dataset.switch_data()
			prob, pred = self.get_memorybank(True)
			ssets = pred.nonzero()[0]
			self.train_data_module.train_dataset.adjust_base_indx(ssets)
			print("change train loader")

		if self.args.use_self_filter_w and self.trainer.current_epoch >= self.args.start_epoch:
			self.train_data_module.train_dataset.switch_data()
			prob, pred = self.get_memorybank_w(True)
			ssets = pred.nonzero()[0]
			self.train_data_module.train_dataset.adjust_base_indx(ssets)
			print("change train loader")




	def validation_step(self, batch, batch_idx):
		# validation_step defines the validation loop.
		# it is independent of forward
		x, y, _ = batch
		feature, y_hat = self(x)
		loss = nn.CrossEntropyLoss(reduction='none')(y_hat, y)
		loss = loss.mean()
		# top1, top5, top10 = accuracy(x_hat, y)
		# # Logging to TensorBoard by default
		self.log("validation_loss", loss)
		# self.log("validation_top1acc", top1)
		# self.log("validation_top5acc", top5)
		# self.log("validation_top10acc", top10)
		# return {'loss':loss, 'top1-acc':top1, 'top5-acc':top5, 'top10-acc':top10}
		return {'loss':loss, 'preds': y_hat, 'targets': y, 'features': feature}

	def validation_epoch_end(self, validation_step_outputs):
		# all_preds = torch.cat([validation_step_outputs[i]['preds'] for i in range(len(validation_step_outputs))])
		# all_targets = torch.cat([validation_step_outputs[i]['targets']for i in range(len(validation_step_outputs))])
		all_preds = torch.cat([validation_step_outputs[i]['preds'] for i in range(len(validation_step_outputs))])
		all_targets = torch.cat([validation_step_outputs[i]['targets']for i in range(len(validation_step_outputs))])
		# all_targets = pl.utilities.distributed.all_gather_ddp_if_available(all_targets)
		all_targets = all_targets.reshape((1,-1))
		# all_preds = pl.utilities.distributed.all_gather_ddp_if_available(all_preds)
		all_preds = all_preds.reshape(-1, all_preds.shape[-1])
		all_preds = all_preds.detach().cpu().numpy()
		all_targets = all_targets.detach().cpu().numpy()
		all_targets = np.squeeze(all_targets)
		top1, top5, top10 = avg_accuracy(all_preds, all_targets)
		_, pred = torch.Tensor(all_preds).topk(1, 1, True, True)
		pred = np.squeeze(pred.detach().cpu().numpy())
		cm = confusion_matrix(all_targets, pred)
		# cls_list = list(set(all_targets))
		# print(cls_list)
		# df_cm = pd.DataFrame(cm/ np.sum(cm, axis=1)[:, None], index=[i for i in cls_list],
		# 				 columns=[i for i in cls_list])
		# Create Heatmap
		plt.figure(figsize=(12, 7))    
		#cm_fig = sn.heatmap(df_cm, annot=True).get_figure()
		cm_fig = sn.heatmap(cm).get_figure()
		if top1 > self.best_acc:
			self.best_acc = top1
			model_path = os.path.join('saved/', self.args.name+'_epoch:'+str(self.trainer.current_epoch)+'_acc:'+str(top1)+'.pt')
			torch.save(self.state_dict(), model_path)
			print("saved model!")

		tensorboard = self.logger.experiment
		tensorboard.add_figure("Confusion matrix", cm_fig, self.trainer.current_epoch)
		self.log("validation_top1acc", top1)
		self.log("validation_top5acc", top5)
		self.log("validation_top10acc", top10)

	def test_step(self, batch, batch_idx):
		# validation_step defines the validation loop.
		# it is independent of forward
		x, y, _ = batch
		feature, y_hat = self(x)
		loss = nn.CrossEntropyLoss(reduction='none')(y_hat, y)
		loss = loss.mean()
		# top1, top5, top10 = accuracy(x_hat, y)
		# # Logging to TensorBoard by default
		self.log("test_loss", loss)
		# self.log("test_top1acc", top1)
		# self.log("test_top5acc", top5)
		# self.log("test_top10acc", top10)
		# return {'loss':loss, 'top1-acc':top1, 'top5-acc':top5, 'top10-acc':top10}
		return {'loss':loss, 'preds': y_hat, 'targets': y, 'features': feature}

	def test_epoch_end(self, test_step_outputs):
		# all_preds = torch.cat([test_step_outputs[i]['preds'] for i in range(len(test_step_outputs))])
		# all_targets = torch.cat([test_step_outputs[i]['targets']for i in range(len(test_step_outputs))])
		# _, pred = all_preds.topk(1, 1, True, True)
		# pred = np.squeeze(pred.detach().cpu().numpy())
		# cm = confusion_matrix(all_targets.detach().cpu().numpy(), pred)
		# print(cm)
		# top1, top5, top10 = avg_accuracy(all_preds, all_targets)
		# self.log("test_top1acc", top1)
		# self.log("test_top5acc", top5)
		# self.log("test_top10acc", top10)
		all_preds = torch.cat([test_step_outputs[i]['preds'] for i in range(len(test_step_outputs))])
		all_targets = torch.cat([test_step_outputs[i]['targets']for i in range(len(test_step_outputs))])
		# all_targets = pl.utilities.distributed.all_gather_ddp_if_available(all_targets)
		all_targets = all_targets.reshape((1,-1))
		# all_preds = pl.utilities.distributed.all_gather_ddp_if_available(all_preds)
		all_preds = all_preds.reshape(-1, all_preds.shape[-1])
		all_preds = all_preds.detach().cpu().numpy()
		all_targets = all_targets.detach().cpu().numpy()
		all_targets = np.squeeze(all_targets)
		top1, top5, top10 = avg_accuracy_with_idx(all_preds, all_targets, self.num_classes)
		_, pred = torch.Tensor(all_preds).topk(1, 1, True, True)
		pred = np.squeeze(pred.detach().cpu().numpy())
		cm = confusion_matrix(all_targets, pred)
		plt.figure(figsize=(12, 7))    
		cm_fig = sn.heatmap(cm).get_figure()
		tensorboard = self.logger.experiment
		tensorboard.add_figure("TEST Confusion matrix", cm_fig, self.trainer.current_epoch)
		# print("validation cm:", cm)
		self.log("test_top1acc", top1)
		self.log("test_top5acc", top5)
		self.log("test_top10acc", top10)


	def configure_optimizers(self):
		optimizer = optim.SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=1e-4)
		return optimizer

