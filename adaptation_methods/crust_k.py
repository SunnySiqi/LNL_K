import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.distributions as dd
from sklearn.metrics import pairwise_distances
import heapq
from torch.autograd import grad
from tqdm import tqdm
import torch.nn.functional as F
from lazyGreedy import lazy_greedy_heap
from fl_cifar import FacilityLocationCIFAR
from LitCNN import *

class crust_LitCNN(LitCNN):
	def __init__(self, args, num_classes=1, train_data_module=None):
		super().__init__(self, args, num_classes=1, train_data_module=None)
		self.control_label = 0
		self.all_targets = None

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

	def estimate_grads_w_noise_knowledge(self):
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
			ctl_label = torch.tensor(np.repeat(self.control_label, len(imgs)))
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

	def crust(self):
		grads_all, labels, indexes = self.estimate_grads()
			# per-class clustering
		ssets = []
		coreset_cls_range = list(set(self.all_targets)-{self.control_label})
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
		return ssets

	def crust_k(self):
		grads_all_dict, labels, indexes = self.estimate_grads_w_noise_knowledge()
			# per-class clustering
		ssets = []
		coreset_cls_range = list(set(self.all_targets)-{self.control_label})
		ctl_ids = np.where((labels == self.control_label) == True)[0]
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
		return ssets


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
		self.all_targets = all_targets
		self.train_data_module.train_dataset.switch_data()   

		if self.args.use_crust and self.trainer.current_epoch >= self.args.start_epoch:
			ssets = self.crust()
		elif self.args.use_crust_w and self.trainer.current_epoch >= self.args.start_epoch:
			ssets = self.crust_k()

		self.train_data_module.train_dataset.adjust_base_indx(ssets)
		print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
		print("change train loader")

