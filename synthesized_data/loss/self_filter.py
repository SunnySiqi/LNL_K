import torch
import torch.nn.functional as F
import math
import torch.nn as nn

__all__ = ['SelfFilterLoss']

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
	
	# def forward(self, output, target, mode=None):
	# 	bs = output.size()[0]
	# 	pseudo_label = torch.softmax(output, dim=1)
	# 	values, index = pseudo_label.topk(k=2, dim=1)
	# 	first_label = index[:, 0]
	# 	second_label = index[:, 1]
	# 	latent_labels = torch.zeros((bs, self.num_classes)).cuda()

	# 	for i in range(bs):
	# 		if target[i] >= 5:
	# 			latent_labels[i] = target[i]
	# 		elif target[i] < 5 and first_label[i] >=5 and second_label[i] >= 5:
	# 			latent_labels[i] = first_label[i]
	# 		else:
	# 			latent_labels[i] = target[i]
	# 	loss = F.cross_entropy(output, latent_labels)
	# 	# if mode == 'warm_up':
	# 	# 	for i in range(bs):
	# 	# 		if target[i] >= 5 or first_label[i] == target[i]:
	# 	# 			latent_labels[i] = target[i]
	# 	# 		elif target[i] < 5 and first_label[i] >=5:
	# 	# 			latent_labels[i] = first_label[i]
	# 	# 		elif target[i] < 5 and first_label[i] < 5:
	# 	# 			latent_labels[i] = target[i]
	# 	# 	loss = F.cross_entropy(output, latent_labels)
	# 	# else:
	# 	# 	for i in range(bs):
	# 	# 		confident = pseudo_label[i]
	# 	# 		max_ = confident[target[i]]
	# 	# 		confident = 0.2 - confident / max_
	# 	# 		mask = (confident >= 0.0)
	# 	# 		latent_labels[i] = confident * mask
	# 	# 		conf_penalty = 0.1 * self.one_lossF(output, latent_labels) / (self.num_classes - 1)
	# 	# 	loss = F.cross_entropy(output, target) + conf_penalty

	# 	return loss