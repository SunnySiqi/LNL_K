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
from tqdm import tqdm
import seaborn as sn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def accuracy(output, target):
	'''
	Computes the accuracy over the k top predictions for the specified values of k
	'''
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
	return [acc_top1.mean(), acc_top5.mean(), acc_top10.mean()]

#define the LightningModule
class LitCNN(pl.LightningModule):
	def __init__(self, args, num_classes=1, train_data_module=None):
		super().__init__()
		self.num_classes = num_classes
		self.args = args
		self.best_acc = 0.0
		self.train_data_module = train_data_module
		net = models.efficientnet_b0(weights='DEFAULT')
		layers = list(net.children())
		#replace the first layer with 5-channel input
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

	@abstractmethod
	def fine(self):
		"""
		Select clean samples to update the train_data_loader with FINE method.
		"""

	def fine_k(self):
		"""
		Select clean samples to update the train_data_loader with FINE+k method.
		"""

	def estimate_grads(self):
		"""
		Helper function of CRUST method.
		"""

	def estimate_grads_w_noise_knowledge(self):
		"""
		Helper function of CRUST+k method.
		"""

	def crust(self):
		"""
		Select clean samples to update the train_data_loader with CRUST method.
		"""

	def crust_k(self):
		"""
		Select clean samples to update the train_data_loader with CRUST+K method.
		"""

	def get_memorybank(self):
		"""
		Helper function of SFT method.
		"""

	def get_memorybank_k(self):
		"""
		Helper function of SFT+K method.
		"""

	def sft(self):
		"""
		Select clean samples to update the train_data_loader with SFT method.
		"""

	def sft_k(self):
		"""
		Select clean samples to update the train_data_loader with SFT+K method.
		"""

		
	def forward(self, x):
		z = self.net(x)
		feature = z.view(z.size(0), -1)
		x_hat = self.linear(feature)
		return feature, x_hat

	def predict_to_select(self):
		'''
		Use the prediction
		'''
		all_features = []
		all_targets = []
		all_indexes = []
		data_loader = self.train_data_module.predict_dataloader()
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

	def training_step(self, batch, batch_idx):
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

		self.log("train_loss", loss)
		return {'loss':loss, 'preds': y_hat, 'targets': y, 'features': feature}

	def training_epoch_end(self, training_step_outputs):
		'''
		Overwritten by subclasses for sample selection.
		'''
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


	def validation_step(self, batch, batch_idx):
		x, y, _ = batch
		feature, y_hat = self(x)
		loss = nn.CrossEntropyLoss(reduction='none')(y_hat, y)
		loss = loss.mean()
		self.log("validation_loss", loss)
		return {'loss':loss, 'preds': y_hat, 'targets': y, 'features': feature}

	def validation_epoch_end(self, validation_step_outputs):
		all_preds = torch.cat([validation_step_outputs[i]['preds'] for i in range(len(validation_step_outputs))])
		all_targets = torch.cat([validation_step_outputs[i]['targets']for i in range(len(validation_step_outputs))])
		all_targets = all_targets.reshape((1,-1))
		all_preds = all_preds.reshape(-1, all_preds.shape[-1])
		all_preds = all_preds.detach().cpu().numpy()
		all_targets = all_targets.detach().cpu().numpy()
		all_targets = np.squeeze(all_targets)
		top1, top5, top10 = avg_accuracy(all_preds, all_targets)
		_, pred = torch.Tensor(all_preds).topk(1, 1, True, True)
		pred = np.squeeze(pred.detach().cpu().numpy())
		cm = confusion_matrix(all_targets, pred)
		plt.figure(figsize=(12, 7))    
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
		x, y, _ = batch
		feature, y_hat = self(x)
		loss = nn.CrossEntropyLoss(reduction='none')(y_hat, y)
		loss = loss.mean()
		self.log("test_loss", loss)
		return {'loss':loss, 'preds': y_hat, 'targets': y, 'features': feature}

	def test_epoch_end(self, test_step_outputs):
		all_preds = torch.cat([test_step_outputs[i]['preds'] for i in range(len(test_step_outputs))])
		all_targets = torch.cat([test_step_outputs[i]['targets']for i in range(len(test_step_outputs))])
		all_targets = all_targets.reshape((1,-1))
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
		self.log("test_top1acc", top1)
		self.log("test_top5acc", top5)
		self.log("test_top10acc", top10)


	def configure_optimizers(self):
		optimizer = optim.SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=1e-4)
		return optimizer

