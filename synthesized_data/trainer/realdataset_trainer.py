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
import copy

from selection.svd_classifier import *
from selection.gmm import *
from selection.util import *
import data_loader.data_loaders as module_data

class NegEntropy(object):
	def __call__(self,outputs):
		probs = torch.softmax(outputs, dim=1)
		return torch.mean(torch.sum(probs.log()*probs, dim=1))

class RealDatasetTrainer(BaseTrainer):
	"""
	DefaultTrainer class

	Note:
		Inherited from BaseTrainer.
	"""
	def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader, parse,
				 valid_data_loader=None, test_data_loader=None, teacher=None, lr_scheduler=None, len_epoch=None, val_criterion=None, mode=None, entropy=False, threshold = 0.1):
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
		self.train_data_loader = data_loader
		self.dynamic_train_data_loader = copy.deepcopy(data_loader)
		self.valid_data_loader = valid_data_loader
		
		self.warm_up = parse.warmup
		self.every = parse.every
		
		self.orig_data_loader = getattr(module_data, self.config['data_loader']['type'])(
			self.config['data_loader']['args']['data_dir'],
			batch_size=self.config['data_loader']['args']['batch_size'],
			shuffle=False,
			validation_split=0.0,
			num_batches=self.config['data_loader']['args']['num_batches'],
			training=True,
			num_workers=self.config['data_loader']['args']['num_workers'],
			pin_memory=self.config['data_loader']['args']['pin_memory'])

		self.test_data_loader = test_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.do_test = self.test_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.log_step = int(np.sqrt(data_loader.batch_size))
		self.train_loss_list: List[float] = []
		self.val_loss_list: List[float] = []
		self.test_loss_list: List[float] = []
		self.teacher_idx = None
		self.easy_idx = None
		self.hard_idx = None
			
		# Visdom visualization
		
		self.entropy = entropy
		if self.entropy: self.entro_loss = Entropy(threshold)

		self.noise_source_dict = {}
		confusing_pairs = self.config['trainer']['confusing_pairs']
		for p in confusing_pairs:
			self.noise_source_dict[int(p[0])] = np.array([int(p[1])])
			self.noise_source_dict[int(p[1])] = np.array([int(p[0])])
		#self.clean_classes = np.array(list(set(np.arange(self.config['num_classes'])) - set(np.array(list(self.noise_source_dict.keys())))))
		self.clean_classes = None
		self.vector_dict = {}
		
	def _eval_metrics(self, output, label):
		acc_metrics = np.zeros(len(self.metrics))
		for i, metric in enumerate(self.metrics):
			acc_metrics[i] += metric(output, label)
			self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
		return acc_metrics

	def update_dataloader(self, epoch):
		
		with torch.no_grad():
			
			self.orig_data_loader = getattr(module_data, self.config['data_loader']['type'])(
				self.config['data_loader']['args']['data_dir'],
				batch_size=self.config['data_loader']['args']['batch_size'],
				shuffle=False,
				validation_split=0.0,
				num_batches=self.config['data_loader']['args']['num_batches'],
				training=True,
				num_workers=self.config['data_loader']['args']['num_workers'],
				pin_memory=self.config['data_loader']['args']['pin_memory']
			)
			
			current_features, current_labels = get_features(self.model, self.orig_data_loader)
			datanum = len(current_labels)
			prev_features, prev_labels = current_features, current_labels
				
			if self.config['subset_training']['fine_with_source']:
				# if epoch < self.config['subset_training']['clean_epoch']:
				#     self.vector_dict, self.teacher_idx = fine_w_noise_source(self.vector_dict, self.noise_source_dict, True, self.clean_classes, current_features, current_labels, fit=self.parse.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=self.parse.zeta)
				# else:
				#     self.vector_dict, self.teacher_idx = fine_w_noise_source(self.vector_dict, self.noise_source_dict, False, self.clean_classes, current_features, current_labels, fit=self.parse.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=self.parse.zeta) 
				self.vector_dict, self.easy_idx, self.teacher_idx, self.hard_idx = fine_w_noise_source(self.vector_dict, self.noise_source_dict, self.clean_classes, current_features, current_labels, fit=self.parse.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=self.parse.zeta)  
			else:
				self.vector_dict, self.teacher_idx = fine(self.vector_dict, current_features, current_labels, fit=self.parse.distill_mode, prev_features=prev_features, prev_labels=prev_labels, p_threshold=self.parse.zeta)
			
		curr_data_loader = getattr(module_data, self.config['data_loader']['type'])(
			self.config['data_loader']['args']['data_dir'],
			batch_size=self.config['data_loader']['args']['batch_size'],
			shuffle=self.config['data_loader']['args']['shuffle'],
			validation_split=0.0,
			num_batches=self.config['data_loader']['args']['num_batches'],
			training=True,
			num_workers=self.config['data_loader']['args']['num_workers'],
			pin_memory=self.config['data_loader']['args']['pin_memory'],
			teacher_idx=self.teacher_idx,
			easy_idx = self.easy_idx,
			hard_idx = self.hard_idx
		)
		
		self.selected, self.precision, self.recall, self.f1, self.specificity, self.accuracy = return_statistics(self.orig_data_loader, self.teacher_idx)
		
		return curr_data_loader
	
	
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

		if epoch > max(self.warm_up, 1) and epoch % self.every == 0:
			self.dynamic_train_data_loader = self.update_dataloader(epoch)
			self.len_epoch = len(self.dynamic_train_data_loader)
			print ('############# Epoch:{} ############'.format(epoch))

		self.model.train()
		
		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		total_metrics_gt = np.zeros(len(self.metrics))
		conf_penalty = NegEntropy()


		with tqdm(self.dynamic_train_data_loader) as progress:
			for batch_idx, (data, label, indexs, gt) in enumerate(progress):
				progress.set_description_str(f'Train epoch {epoch}')
				
				data, label = data.to(self.device), label.long().to(self.device)
				gt = gt.long().to(self.device)
				
				_, output = self.model(data)
				loss = self.train_criterion(output, label, indexs)
#                 if epoch < 3:
#                 loss += conf_penalty(output)

				self.optimizer.zero_grad()
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
			'num_train': len(self.dynamic_train_data_loader.dataset),
			'metrics': (total_metrics / self.len_epoch).tolist(),
			'metrics_gt': (total_metrics_gt / self.len_epoch).tolist(),
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
		with torch.no_grad():
			with tqdm(self.valid_data_loader) as progress:
				for batch_idx, (data, label, _, _) in enumerate(progress):
					progress.set_description_str(f'Valid epoch {epoch}')
					data, label = data.to(self.device), label.to(self.device)
					_, output = self.model(data)
					loss = self.val_criterion()(output, label)

					self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
					self.writer.add_scalar('loss', loss.item())
					self.val_loss_list.append(loss.item())
					total_val_loss += loss.item()
					total_val_metrics += self._eval_metrics(output, label)
					self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

		# add histogram of model parameters to the tensorboard
		for name, p in self.model.named_parameters():
			self.writer.add_histogram(name, p, bins='auto')

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
		results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
		tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
		with torch.no_grad():
			with tqdm(self.test_data_loader) as progress:
				for batch_idx, (data, label, indexs, _) in enumerate(progress):
					progress.set_description_str(f'Test epoch {epoch}')
					data, label = data.to(self.device), label.to(self.device)
					_, output = self.model(data)
					
					loss = self.val_criterion()(output, label)

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

		return {
			'test_loss': total_test_loss / len(self.test_data_loader),
			'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
		},[results,tar_]


	def _warmup_epoch(self, epoch):
		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		self.model.train()
		data_loader = self.data_loader#self.loader.run('warmup')


		with tqdm(data_loader) as progress:
			for batch_idx, (data, label, indexs , _) in enumerate(progress):
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