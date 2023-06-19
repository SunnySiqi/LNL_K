import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import heapq
from torch.autograd import grad
from tqdm import tqdm
import torch.nn.functional as F
from sft_loss import SelfFilterLoss
from LitCNN import *

class sft_LitCNN(LitCNN):
	def __init__(self, args, num_classes=1, train_data_module=None):
		super().__init__(self, args, num_classes=1, train_data_module=None)
		self.memory_bank = []
		self.self_filter_loss = SelfFilterLoss(self.num_classes)
		self.selection = False

	def get_memorybank(self):
		k = int(self.args.self_filter_k)

		if self.selection:
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
		if self.selection:
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

	def get_memorybank_k(self):
		k = int(self.args.self_filter_k)

		if self.selection:
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
		if self.selection:
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

	def sft(self):
		self.selection = True
		prob, pred = self.get_memorybank()
		ssets = pred.nonzero()[0]
		return ssets

	def sft_k(self):
		self.selection = True
		prob, pred = self.get_memorybank_k()
		ssets = pred.nonzero()[0]
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

		if self.trainer.current_epoch < self.args.start_epoch: 
			if self.args.use_self_filter:
				self.get_memorybank(False)
			elif self.args.use_self_filter_w:
				self.get_memorybank_k(False)

		else:
			self.train_data_module.train_dataset.switch_data()
			if self.args.use_self_filter:
				ssets = self.sft()
			elif self.args.use_self_filter_w:
				ssets = self.sft_k()
			self.train_data_module.train_dataset.adjust_base_indx(ssets)
			print("NEW DATA!!!!!!", str(self.trainer.global_rank) + "_" + str(len(self.train_data_module.train_dataset.train_imnames)))
			print("change train loader")
