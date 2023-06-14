import numpy as np
import torch
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils.util import inf_loop
import sys
from sklearn.mixture import GaussianMixture
import pdb
import numpy as np
from torch.autograd import grad
from sklearn.metrics import confusion_matrix



class DefaultTrainer(BaseTrainer):
	"""
	DefaultTrainer class

	Note:
		Inherited from BaseTrainer.
	"""
	def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader, parse,
				 valid_data_loader=None, test_data_loader=None, teacher = None, lr_scheduler=None, len_epoch=None, val_criterion=None, mode=None, entropy=False, threshold = 0.1):
		super().__init__(model, train_criterion, metrics, optimizer, config, val_criterion, parse)
		self.config = config
		self.data_loader = data_loader
		self.mode = mode
		if len_epoch is None:
			# epoch-based training
			self.len_epoch = len(self.data_loader)
		else:
			# iteration-based training
			self.data_loader = inf_loop(data_loader)
			self.len_epoch = len_epoch
		self.valid_data_loader = valid_data_loader
		
		if teacher != None:
			self.teacher = teacher.to(self.device)
		else:
			self.teacher = teacher

		self.test_data_loader = test_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.do_test = self.test_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.log_step = int(np.sqrt(data_loader.batch_size))
		self.train_loss_list: List[float] = []
		self.val_loss_list: List[float] = []
		self.test_loss_list: List[float] = []
		self.purity = (data_loader.train_dataset.train_labels == \
					   data_loader.train_dataset.train_labels_gt).sum() / len(data_loader.train_dataset)
	
		# Visdom visualization
		self.entropy = entropy
		if self.entropy:
			self.entro_loss = Entropy(threshold)

		if self.config['trainer']['get_noise_source'] > 0:
			self.noise_source_dict = {}
			self.clean_classes = np.array([])
			self.noise_sources = set()
		else:
			self.noise_source_dict = {}
			if self.config['trainer']['control']:
				ctl_cls_lower_bound = self.config['trainer']['ctl_cls_lower_bound']
				c_cls_range = range(ctl_cls_lower_bound)
				for c in c_cls_range:
					self.noise_source_dict[c] = np.arange(ctl_cls_lower_bound, self.config['num_classes'])
				self.clean_classes = np.array(list(set(np.arange(self.config['num_classes'])) - set(np.array(list(self.noise_source_dict.keys())))))
			else:
				confusing_pairs = self.config['trainer']['asym_pairs']
				for p in confusing_pairs:
					# self.noise_source_dict[int(p[0])] = np.array([int(p[1])])
					self.noise_source_dict[int(p[1])] = np.array([int(p[0])])
				self.clean_classes =  np.array(list(set(np.arange(self.config['num_classes'])) - set(np.array(list(self.noise_source_dict.keys())))))
			self.noise_sources = set()
			for target_class in self.noise_source_dict:
				for noise_class in self.noise_source_dict[target_class]:
					self.noise_sources.add(noise_class)

	def _eval_metrics(self, output, label):
		acc_metrics = np.zeros(len(self.metrics))
		for i, metric in enumerate(self.metrics):
			acc_metrics[i] += metric(output, label)
			self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
		return acc_metrics


	def estimate_grads(self, epoch):
	# switch to train mode
		self.model.train()
		all_grads = []
		all_labels = []
		all_indexs = []
		with tqdm(self.data_loader) as progress:
			for batch_idx, (data, label, indexs, _) in enumerate(progress):
				data = data.to(self.device)
				all_labels.append(label)
				all_indexs.append(indexs)
				label = label.to(self.device)
				# compute output
				feat, output = self.model(data)
				if self.config['train_loss']['type'] == 'CrossEntropyLoss':
					loss = self.train_criterion()(output, label)
				else:
					loss = self.train_criterion(indexs.cpu().detach().numpy().tolist(), output, label)
				est_grad = grad(loss, feat)
				all_grads.append(est_grad[0].detach().cpu().numpy())
			all_grads = np.vstack(all_grads)
			all_labels = np.hstack(all_labels)
			all_indexs = np.hstack(all_indexs)
		return all_grads, all_labels, all_indexs

	def estimate_grads_w_noise_knowledge(self, epoch):
	# switch to train mode
		self.model.train()
		all_grads_dict = {}
		all_grads_dict['self'] = []
		for noise_class in list(self.noise_sources):
			all_grads_dict[noise_class] = []
		all_labels = []
		all_indexs = []

		with tqdm(self.data_loader) as progress:
			for batch_idx, (data, label, indexs, _) in enumerate(progress):
				data = data.to(self.device)
				all_labels.append(label)
				all_indexs.append(indexs)
				label = label.to(self.device)
				# compute output
				feat, output = self.model(data)
				loss = self.train_criterion()(output, label)
				est_grad = grad(loss, feat)
				all_grads_dict['self'].append(est_grad[0].detach().cpu().numpy())

				for noise_class in list(self.noise_sources):
					feat, output = self.model(data)
					noise_label = torch.tensor(np.repeat(noise_class, len(data)))
					noise_label = noise_label.to(self.device)
					loss_noise = self.train_criterion()(output, noise_label)
					est_grad = grad(loss_noise, feat)
					all_grads_dict[noise_class].append(est_grad[0].detach().cpu().numpy())

			for dict_key in all_grads_dict:
				all_grads_dict[dict_key] = np.vstack(all_grads_dict[dict_key])
			all_labels = np.hstack(all_labels)
			all_indexs = np.hstack(all_indexs)
		return all_grads_dict, all_labels, all_indexs


	def get_feature(self, epoch):
		self.model.eval()
		all_labels = []
		all_indexs = []
		all_features = []
		with tqdm(self.data_loader) as progress:
			for batch_idx, (data, label, indexs, _) in enumerate(progress):
				data = data.to(self.device)
				all_labels.append(label)
				all_indexs.append(indexs)
				label = label.to(self.device)
				feat, output = self.model(data)
				all_features.append(feat.detach().cpu().numpy())
		all_labels = np.hstack(all_labels)
		all_indexs = np.hstack(all_indexs)
		all_features = np.concatenate(all_features, axis=0)
		return all_labels, all_features, all_indexs

	def get_gt_samples(self, epoch):
		clean_idx = []
		with tqdm(self.data_loader) as progress:
			for batch_idx, (data, label, indexs, label_gt) in enumerate(progress):
				clean_idx += indexs[list(np.where((label == label_gt) == True)[0])]
		return clean_idx

	def get_memorybank(self, epoch):
		if epoch >= self.config['trainer']['warmup']:
			selection = True
		else:
			selection = False

		k = int(self.config['subset_training']['self_filter_k'])

		self.model.eval()

		if selection:
			memory_bank_last = self.memory_bank[-1]
			fluctuation_ = [0] * 50000

		result = torch.zeros(50000)
		with torch.no_grad():
			with tqdm(self.data_loader) as progress:
				for batch_idx, (data, label, indexs, _) in enumerate(progress):
					data = data.to(self.device)
					label = label.to(self.device)
					feat, output = self.model(data)
					max_probs, pred = output.max(1)

					for b in range(data.size(0)):
						sample_index = indexs[b]
						if pred[b] == label[b]:
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

			confidence_smooth = torch.zeros_like(self.memory_bank[0])
			for i in range(k):
				confidence_smooth += self.memory_bank[i]
			prob = (confidence_smooth.numpy() + np.array(fluctuation_)) / (k + 1)  # adding confidence make fluctuation more smooth
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

	def get_memorybank_w(self, epoch):
		if epoch >= self.config['trainer']['warmup']:
			selection = True
		else:
			selection = False

		k = int(self.config['subset_training']['self_filter_k'])

		self.model.eval()

		if selection:
			memory_bank_last = self.memory_bank[-1]
			fluctuation_ = [0] * 50000

		clean_sset_id = []
		result = torch.zeros(50000)
		sample_class_idx_dict ={}
		with torch.no_grad():
			with tqdm(self.data_loader) as progress:
				for batch_idx, (data, label, indexs, _) in enumerate(progress):
					data = data.to(self.device)
					label = label.to(self.device)
					feat, output = self.model(data)
					max_probs, pred = output.max(1)

					pred = pred.cpu().detach().numpy().tolist()
					label = label.cpu().detach().numpy().tolist()
					indexs = indexs.cpu().detach().numpy().tolist()
					for b in range(data.size(0)):
						if label[b] not in sample_class_idx_dict:
							sample_class_idx_dict[label[b]] = []
						sample_class_idx_dict[label[b]].append(indexs[b])
						
						sample_index = indexs[b]
						if label[b] in self.clean_classes:
							clean_sset_id.append(sample_index)

						if pred[b] == label[b]:
							result[sample_index] = max_probs[b]
						elif label[b] in list(self.noise_source_dict.keys()) and pred[b] in self.noise_source_dict[label[b]]:
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

			confidence_smooth = torch.zeros_like(self.memory_bank[0])
			for i in range(k):
				confidence_smooth += self.memory_bank[i]
			prob = (confidence_smooth.numpy() + np.array(fluctuation_)) / (k + 1)  # adding confidence make fluctuation more smooth
			#pred = (prob > 0.5)
			prob[clean_sset_id] = 1
			pred = (prob > 0.5)
			ssets = pred.nonzero()[0]
			select_sample_class_dict ={}
			for c in range(10):
				select_sample_class_dict[c] = len(list(set(sample_class_idx_dict[c]).intersection(set(ssets))))
			print("Selected Sample Dict!!!!!!!!!!!!!")
			print(select_sample_class_dict)

			return prob, pred

		else:
			if len(self.memory_bank) < k:
				self.memory_bank.append(result)
			else:
				for i in range(k - 1):
					self.memory_bank[i] = self.memory_bank[i + 1]
				self.memory_bank[-1] = result
			return


	def _train_epoch(self, epoch):
		"""

		:param epoch: Current training epoch.
		:return: A log that contains all information you want to save.

		Note:
			If you have additional information to record, for example:
				> additional_log = {"x": x, "y": y}
			merge it with log before return. i.e.
				> log = {**log, **additional_log}
				> return log

			The metrics in log must have the key 'metrics'.
		"""
		
		self.model.train()

		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		total_metrics_gt = np.zeros(len(self.metrics))

		with tqdm(self.data_loader) as progress:
			for batch_idx, (data, label, indexs, gt) in enumerate(progress):
				progress.set_description_str(f'Train epoch {epoch}')
				
				data, label = data.to(self.device), label.long().to(self.device)
				if self.teacher:
					tea_represent, tea_logit = self.teacher(data)
					tea_represent, tea_logit = tea_represent.to(self.device), tea_logit.to(self.device)
#                     represent_out = self.represent(data).to(self.device)
					
				
				gt = gt.long().to(self.device)
				
				model_represent, output = self.model(data)
				if self.config['train_loss']['type'] == 'CLoss' or self.config['train_loss']['type'] == 'NPCLoss':
					loss = self.train_criterion(output, label, epoch, indexs.cpu().detach().numpy().tolist())
				else:
					if self.teacher:
						loss = self.train_criterion(output, label, indexs, mode=self.mode)
					else:
						sing_lbl = None
						if self.config['train_loss']['type'] == 'CrossEntropyLoss':
							loss = self.train_criterion()(output, label)
						elif self.config['train_loss']['type'] == 'SelfFilterLoss':
							if epoch <= self.config['trainer']['warmup']:
								loss = self.train_criterion(output, label, 'warm_up')
							else:
								loss = self.train_criterion(output, label, 'train')
						else:
							loss = self.train_criterion(output, label, indexs.cpu().detach().numpy().tolist())
#                 pdb.set_trace()
				self.optimizer.zero_grad()
				if self.entropy:
					loss -= self.entro_loss(output, label, sing_lbl)
				loss.backward()

				self.optimizer.step()

				self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
				self.writer.add_scalar('loss', loss.item())
				self.train_loss_list.append(loss.item())
				total_loss += loss.item()
				total_metrics += self._eval_metrics(output, label)
				total_metrics_gt += self._eval_metrics(output, gt)

				if batch_idx % self.log_step == 0:
					progress.set_postfix_str(' {} Loss: {:.6f}'.format(
						self._progress(batch_idx),
						loss.item()))
					self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

				if batch_idx == self.len_epoch:
					break
		# if hasattr(self.data_loader, 'run'):
		#     self.data_loader.run()
		log = {
			'loss': total_loss / self.len_epoch,
			'metrics': (total_metrics / self.len_epoch).tolist(),
			'metrics_gt': (total_metrics_gt / self.len_epoch).tolist(),
			'learning rate': self.lr_scheduler.get_lr(),
			'purity:': '{} = {}/{}'.format(self.purity, (self.data_loader.train_dataset.train_labels == \
				   self.data_loader.train_dataset.train_labels_gt).sum(), len(self.data_loader.train_dataset))
		}


		if self.do_validation:
			val_log = self._valid_epoch(epoch)
			log.update(val_log)
		if self.do_test:
			test_log, test_meta = self._test_epoch(epoch)
			log.update(test_log)
		else: 
			test_meta = [0,0]


		if self.lr_scheduler is not None:
			self.lr_scheduler.step()
			
		return log


	def _valid_epoch(self, epoch):
		"""
		Validate after training an epoch

		:return: A log that contains information about validation

		Note:
			The validation metrics in log must have the key 'val_metrics'.
		"""
		self.model.eval()

		total_val_loss = 0
		total_val_metrics = np.zeros(len(self.metrics))
		all_labels = []
		all_preds = []
		with torch.no_grad():
			with tqdm(self.valid_data_loader) as progress:
				for batch_idx, (data, label, index, _) in enumerate(progress):
					progress.set_description_str(f'Valid epoch {epoch}')
					data, label = data.to(self.device), label.to(self.device)
					_, output = self.model(data)
					loss = self.val_criterion()(output, label)
					all_labels += label.cpu().detach().numpy().tolist()
					_, y_pred = output.max(1)
					all_preds += y_pred.cpu().detach().numpy().tolist()

					self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
					self.writer.add_scalar('loss', loss.item())
					self.val_loss_list.append(loss.item())
					total_val_loss += loss.item()
					total_val_metrics += self._eval_metrics(output, label)
					self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

		# add histogram of model parameters to the tensorboard
		for name, p in self.model.named_parameters():
			self.writer.add_histogram(name, p, bins='auto')
		print("!!!!!VALIDATION CONFUSION MATRIX!!!!!")
		print(confusion_matrix(all_labels, all_preds))
		print("-------------------------------------")

		return {
			'val_loss': total_val_loss / len(self.valid_data_loader),
			'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
		}

	def _test_epoch(self, epoch):
		"""
		Test after training an epoch

		:return: A log that contains information about test

		Note:
			The Test metrics in log must have the key 'val_metrics'.
		"""
		self.model.eval()
		total_test_loss = 0
		total_test_metrics = np.zeros(len(self.metrics))
		all_labels = []
		all_preds = []
		results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
		tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
		with torch.no_grad():
			with tqdm(self.test_data_loader) as progress:
				for batch_idx, (data, label,indexs,_) in enumerate(progress):
					progress.set_description_str(f'Test epoch {epoch}')
					data, label = data.to(self.device), label.to(self.device)
					_, output = self.model(data)
					
					loss = self.val_criterion()(output, label)

					all_labels += label.cpu().detach().numpy().tolist()
					_, y_pred = output.max(1)
					all_preds += y_pred.cpu().detach().numpy().tolist()

					self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
					self.writer.add_scalar('loss', loss.item())
					self.test_loss_list.append(loss.item())
					total_test_loss += loss.item()
					total_test_metrics += self._eval_metrics(output, label)
					self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

					results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
					tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()

		# add histogram of model parameters to the tensorboard
		for name, p in self.model.named_parameters():
			self.writer.add_histogram(name, p, bins='auto')

		print("!!!!!TEST CONFUSION MATRIX!!!!!")
		print(confusion_matrix(all_labels, all_preds))
		print("-------------------------------------")

		return {
			'test_loss': total_test_loss / len(self.test_data_loader),
			'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
		},[results,tar_]


	def _warmup_epoch(self, epoch):
		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		self.model.train()

		with tqdm(self.data_loader) as progress:
			for batch_idx, (data, label, indexs, gt) in enumerate(progress):
				progress.set_description_str(f'Warm up epoch {epoch}')

				data, label = data.to(self.device), label.long().to(self.device)

				self.optimizer.zero_grad()
				_, output = self.model(data)
				out_prob = torch.nn.functional.softmax(output).data.detach()

				#self.train_criterion.update_hist(indexs.cpu().detach().numpy().tolist(), out_prob)

				loss = torch.nn.functional.cross_entropy(output, label)

				loss.backward() 
				self.optimizer.step()

				self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
				self.writer.add_scalar('loss', loss.item())
				self.train_loss_list.append(loss.item())
				total_loss += loss.item()
				total_metrics += self._eval_metrics(output, label)


				if batch_idx % self.log_step == 0:
					progress.set_postfix_str(' {} Loss: {:.6f}'.format(
						self._progress(batch_idx),
						loss.item()))
					self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

				if batch_idx == self.len_epoch:
					break
		if hasattr(self.data_loader, 'run'):
			self.data_loader.run()
		log = {
			'loss': total_loss / self.len_epoch,
			'noise detection rate' : 0.0,
			'metrics': (total_metrics / self.len_epoch).tolist(),
			'learning rate': self.lr_scheduler.get_lr()
		}

		if self.do_validation:
			val_log = self._valid_epoch(epoch)
			log.update(val_log)
		if self.do_test:
			test_log, test_meta = self._test_epoch(epoch)
			log.update(test_log)
		else: 
			test_meta = [0,0]

		return log


	def _progress(self, batch_idx):
		base = '[{}/{} ({:.0f}%)]'
		if hasattr(self.data_loader, 'n_samples'):
			current = batch_idx * self.data_loader.batch_size
			total = self.data_loader.n_samples
		else:
			current = batch_idx
			total = self.len_epoch
		return base.format(current, total, 100.0 * current / total)