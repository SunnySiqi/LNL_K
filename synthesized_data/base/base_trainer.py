from typing import TypeVar, List, Tuple
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import numpy as np
from fl_cifar import FacilityLocationCIFAR
from lazyGreedy import lazy_greedy_heap
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
import torch.distributions as dd
from sklearn.cluster import KMeans

import wandb

def get_distribution(features):
	return dd.normal.Normal(torch.tensor(np.mean(features, axis=0)), torch.tensor(np.std(features, axis=0)+1e-9))


class GrowingCluster(object):
	def __init__(self, startpoint):
		self.points = [startpoint]
		self.centroid = startpoint
	def add_points(self, newpoint):
		self.points.append(newpoint)
		self.centroid = np.mean(self.points, axis=0)

def sset_distribution_init(config, ctl_centroids, c_features):
	if config['subset_training']['naive_init']:
		return get_distribution(c_features)
	if config['subset_training']['naive_centroid_init']:
		startpoint = np.mean(c_features, axis=0)
		c_cluster = GrowingCluster(startpoint)
		candidates = c_features
	else:
		startpoint_idx = np.argmax(np.min(pairwise_distances(c_features, ctl_centroids), axis=1))
		c_cluster = GrowingCluster(c_features[startpoint_idx])
		candidates = np.delete(c_features, startpoint_idx, axis=0)
	targets = ctl_centroids + [c_cluster.centroid]
	c_label = len(targets)-1
	while True:
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
	print("number of points for c_cluster:", len(c_cluster.points))
	return get_distribution(np.array(c_cluster.points))

def get_centroid_sset(centroids, ctl_features, num_ctl, sample_features, sample_ids):
	# ctl_pred = centroids.predict(ctl_features)
	# values, counts = np.unique(ctl_pred, return_counts=True)
	# if len(values) > num_ctl:
	#     ind = np.argpartition(-counts, kth=num_ctl)[:num_ctl]
	#     ctl_label = values[ind]
	#     c_label = list(set(np.arange(num_ctl+1)) - set(ctl_label))[0]
	# else:
	#     c_label = list(set(np.arange(num_ctl+1)) - set(values))[0]
	sample_pred = centroids.predict(sample_features)
	values, counts = np.unique(sample_pred, return_counts=True)
	c_label = values[np.argwhere(counts == np.max(counts))][0][0]
	#sample_pred = centroids.predict(sample_features)
	sset_ids = np.where((sample_pred == c_label) == True)[0]
	sset = sample_ids[np.array(sset_ids)]
	return sset

class BaseTrainer:
	"""
	Base class for all trainers
	"""
	def __init__(self, model, train_criterion, metrics, optimizer, config, val_criterion, parse):
		self.config = config
		self.cls_num = self.config['num_classes']
		self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
		self.parse = parse
		# setup GPU device if available, move model into configured device
		self.device, device_ids = self._prepare_device(config['n_gpu'])
		
		self.model = model.to(self.device) if type(model) is not list else model[0].to(self.device)
#         self.model = model.to(self.device)

		if len(device_ids) > 1:
			self.model = torch.nn.DataParallel(model, device_ids=device_ids)

		if config['train_loss']['type'] == 'CrossEntropyLoss':
			self.train_criterion = train_criterion
		else:
			self.train_criterion = train_criterion.to(self.device)
		
		
		self.val_criterion = val_criterion
		self.metrics = metrics
		
		self.optimizer = optimizer if type(optimizer) is not list else optimizer[0]
#         self.optimizer = optimizer

		cfg_trainer = config['trainer']
		self.epochs = cfg_trainer['epochs']
		self.save_period = cfg_trainer['save_period']
		self.monitor = cfg_trainer.get('monitor', 'off')

		# configuration to monitor model performance and save best
		if self.monitor == 'off':
			self.mnt_mode = 'off'
			self.mnt_best = 0
		else:
			self.mnt_mode, self.mnt_metric = self.monitor.split()
			assert self.mnt_mode in ['min', 'max']

			self.mnt_best = inf if self.mnt_mode == 'min' else -inf
			self.early_stop = cfg_trainer.get('early_stop', inf)

		self.start_epoch = 1

		self.checkpoint_dir = config.save_dir

		# setup visualization writer instance                
		self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

		if config.resume is not None:
			self._resume_checkpoint(config.resume)

		if self.config['subset_training']['self_filter'] or self.config['subset_training']['self_filter_w']:
			self.memory_bank = []

	@abstractmethod
	def _train_epoch(self, epoch):
		"""
		Training logic for an epoch

		:param epoch: Current epochs number
		"""
		raise NotImplementedError

	def estimate_grads(self, epoch):
		"""
		Get the grads for sub-training set selection
		:param epoch: Current epochs number
		"""
		raise NotImplementedError

	def get_feature(self, epoch):
		"""
		Get the features for sub-training set selection
		:param epoch: Current epochs number
		"""

	def get_gt_samples(self, epoch):
		"""
		Get the ground truth clean samples for sub-training set selection
		:param epoch: Current epochs number
		"""

	def get_memorybank(self, epoch):
		"""
		Get the memorybank for sub-training set selection
		:param epoch: Current epochs number
		"""

	def get_memorybank_w(self, epoch):
		"""
		Get the memorybank with noise source dict for sub-training set selection
		:param epoch: Current epochs number
		"""


	def train(self):
		"""
		Full training logic
		"""
		not_improved_count = 0
		if self.config['subset_training']['self_filter'] or self.config['subset_training']['self_filter_w']:
			memory_bank = []

		for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
			if epoch <= self.config['trainer']['warmup']:
				if self.config['subset_training']['self_filter']:
					self.get_memorybank(epoch)
				elif self.config['subset_training']['self_filter_w']:
					self.get_memorybank_w(epoch)

				result = self._warmup_epoch(epoch)

			else:
				if self.config['trainer']['get_noise_source'] > 0 and epoch % self.config['trainer']['get_noise_source'] == 0:
					self.noise_source_dict = {}
					self.data_loader.train_dataset.switch_data()
					all_labels, all_features, _ = self.get_feature(epoch)
					kmeans = KMeans(n_clusters=self.cls_num, random_state=0, n_init=10).fit(all_features)
					for class_id in range(self.cls_num):
						sample_idx = np.where((kmeans.labels_ == class_id) == True)[0]
						labels = all_labels[sample_idx]
						values, counts = np.unique(labels, return_counts=True)
						if len(values) == 1:
							continue
						top2 = np.argpartition(-counts, kth=1)[:2]
						if values[top2][0] not in self.noise_source_dict:
							self.noise_source_dict[values[top2][0]] = set()
						self.noise_source_dict[values[top2][0]].add(values[top2][1])
					self.noise_sources = set()
					for target_class in self.noise_source_dict:
						self.noise_source_dict[target_class] = np.array(list(self.noise_source_dict[target_class]))
						for noise_class in self.noise_source_dict[target_class]:
							self.noise_sources.add(noise_class)
					self.clean_classes = set(np.arange(self.cls_num)).difference(set(list(self.noise_source_dict.keys())))
					print("NOISE SOURCE DICT UPDATED!!!!!!!!!")
					print(self.noise_source_dict)
					print(self.noise_sources)
					print(self.clean_classes)


				if self.config['subset_training']['oracle']:
					self.data_loader.train_dataset.switch_data()
					ssets = self.get_gt_samples(epoch)
					self.data_loader.train_dataset.adjust_base_indx_tmp(ssets)
					#self.data_loader.train_dataset.train_balance_sample()
				elif self.config['subset_training']['use_crust']:
					self.data_loader.train_dataset.switch_data()
					grads_all, labels_all, indexs_all = self.estimate_grads(epoch)
					# per-class clustering
					ssets = []
					for c in list(set(labels_all)):
						sample_ids = np.where((labels_all == c) == True)[0]
						grads = grads_all[sample_ids]
						dists = pairwise_distances(grads)
						V = range(len(grads))
						F = FacilityLocationCIFAR(V, D=dists)
						B = int(self.config['subset_training']['crust_fl_ration'] * len(grads))
						sset, vals = lazy_greedy_heap(F, V, B)
						sset = sample_ids[np.array(sset)]
						ssets += list(sset)
					self.data_loader.train_dataset.adjust_base_indx_tmp(indexs_all[ssets])
					#self.data_loader.train_dataset.train_balance_sample()
					print("change train loader")

				# elif self.config['subset_training']['use_gmm']:
				# 	self.data_loader.train_dataset.switch_data()
				# 	all_labels, all_features = self.get_feature(epoch)
				# 	# per-class clustering
				# 	ssets = []
				# 	ctl_cls_lower_bound = self.config['trainer']['ctl_cls_lower_bound']
				# 	num_ctl = self.config['num_classes']- ctl_cls_lower_bound
				# 	ctl_ids = np.where((all_labels >= ctl_cls_lower_bound) == True)[0]
				# 	ctl_features = all_features[ctl_ids]
				# 	ctl_gm = GaussianMixture(n_components=num_ctl, random_state=0).fit(ctl_features)
				# 	for c in range(ctl_cls_lower_bound):
				# 		sample_ids = np.where((all_labels == c) == True)[0]
				# 		c_feature = all_features[sample_ids]
				# 		gmm_score = ctl_gm.score_samples(c_feature)
				# 		threshold = np.quantile(gmm_score, self.config['subset_training']['gmm_ratio'])
				# 		sset_ids = np.where((gmm_score < threshold) == True)[0]
				# 		sset = sample_ids[np.array(sset_ids)]
				# 		ssets += list(sset)
				# 	for c in range(ctl_cls_lower_bound, self.config['num_classes']):
				# 		sample_ids = np.where((all_labels == c) == True)[0]
				# 		sset = sample_ids
				# 		ssets += list(sset)
				# 	self.data_loader.train_dataset.adjust_base_indx_tmp(ssets)
				# 	print("change train loader")

				elif self.config['subset_training']['adptive_crust']:
					self.data_loader.train_dataset.switch_data()
					grads_all_dict, labels_all, indexs_all = self.estimate_grads_w_noise_knowledge(epoch)
					# per-class clustering
					ssets = []
					for c in list(set(labels_all)):
						sample_ids = np.where((labels_all == c) == True)[0]
						grads_self = grads_all_dict['self'][sample_ids]
						if c not in self.noise_source_dict and c not in self.clean_classes:
							dists = pairwise_distances(grads_self)
							V = range(len(grads_self))
							F = FacilityLocationCIFAR(V, D=dists)
							B = int(self.config['subset_training']['crust_fl_ration'] * len(grads_self))
							sset, vals = lazy_greedy_heap(F, V, B)
							sset = sample_ids[np.array(sset)]
						elif c in self.clean_classes:
							sset = sample_ids
						else:
							sset = sample_ids
							noisy_num_per_class = int((1-self.config['subset_training']['crust_fl_ration'])/len(self.noise_source_dict[c])*len(sset))
							B = noisy_num_per_class + len(sset)
							for noise_source in self.noise_source_dict[c]:
								grad_noise = grads_all_dict[noise_source][sset]
								noise_source_sample_ids = np.where((labels_all == noise_source) == True)[0]
								grad_noise_self = grads_all_dict['self'][noise_source_sample_ids]
								grads_all = np.concatenate((grad_noise, grad_noise_self), axis=0)
								sample_ids_all = np.concatenate((sset, noise_source_sample_ids), axis=0)
								dists = pairwise_distances(grads_all)
								V = range(len(grads_all))
								F = FacilityLocationCIFAR(V, D=dists)
								B = int((1-self.config['subset_training']['crust_fl_ration'])/len(self.noise_source_dict[c]) * len(grad_noise) + len(noise_source_sample_ids))
								# print("NOISE SAMPLES!!!!!!!!!", len(noise_source_sample_ids))
								# print("CLASS SAMPLES!!!!!!!!!!", len(grad_noise))
								noise_sset, vals = lazy_greedy_heap(F, V, B)
								# print("LEN SELECTED!!!!!!!", len(noise_sset))
								noise_sset = sample_ids_all[np.array(noise_sset)]
								top_class_intersection = list(set(sset).intersection(set(noise_sset)))[:noisy_num_per_class]
								# print("CLASS INTERSECTION", len(list(set(sset).intersection(set(noise_sset)))))
								# print("NOISE INTERSECTION", len(list(set(noise_source_sample_ids).intersection(set(noise_sset)))))
								#sset = np.array(list(set(sset).difference(set(noise_sset))))
								sset = np.array(list(set(sset).difference(set(top_class_intersection))))
								sset = sset
						ssets += list(sset)
					self.data_loader.train_dataset.adjust_base_indx_tmp(indexs_all[ssets])
					# self.data_loader.train_dataset.train_balance_sample()
					print("change train loader")
					print("NUMBER OF CLASS 3", (self.data_loader.train_dataset.train_labels == 3).sum())
					print("NUMBER OF CLASS 4", (self.data_loader.train_dataset.train_labels == 4).sum())
					print("NUMBER OF CLASS 9", (self.data_loader.train_dataset.train_labels == 9).sum())
					print("NUMBER OF CLASS 1", (self.data_loader.train_dataset.train_labels == 1).sum())
					print("NUMBER OF CLASS 5", (self.data_loader.train_dataset.train_labels == 5).sum())
					print("NUMBER OF CLASS 7", (self.data_loader.train_dataset.train_labels == 7).sum())
					print("CLEAN NUMBER OF CLASS 1", (self.data_loader.train_dataset.train_labels_gt == 1).sum())
					print("CLEAN NUMBER OF CLASS 5", (self.data_loader.train_dataset.train_labels_gt == 5).sum())
					print("CLEAN NUMBER OF CLASS 7", (self.data_loader.train_dataset.train_labels_gt == 7).sum())


				elif self.config['subset_training']['self_filter']:
					self.data_loader.train_dataset.switch_data()
					prob, pred = self.get_memorybank(epoch)
					ssets = pred.nonzero()[0]
					self.data_loader.train_dataset.adjust_base_indx_tmp(ssets)
					# self.data_loader.train_dataset.train_balance_sample()
					print("change train loader")

				elif self.config['subset_training']['self_filter_w']:
					self.data_loader.train_dataset.switch_data()
					prob, pred = self.get_memorybank_w(epoch)
					ssets = pred.nonzero()[0]
					self.data_loader.train_dataset.adjust_base_indx_tmp(ssets)
					# self.data_loader.train_dataset.train_balance_sample()
					print("change train loader")


				elif self.config['subset_training']['use_class_gmm']:
					self.data_loader.train_dataset.switch_data()
					all_labels, all_features, indexs_all = self.get_feature(epoch)
					# per-class clustering
					ssets = []
					if self.config['trainer']['control']:
						ctl_cls_lower_bound = self.config['trainer']['ctl_cls_lower_bound']
						num_ctl = self.config['num_classes']- ctl_cls_lower_bound
						ctl_cls_range = range(ctl_cls_lower_bound, self.config['num_classes'])
						c_cls_range = range(ctl_cls_lower_bound)
						ctl_distributions = [get_distribution(all_features[np.where((all_labels == c) == True)[0]]) for c in ctl_cls_range]
						ctl_centroids = [np.mean(all_features[np.where((all_labels == c) == True)[0]], axis=0) for c in ctl_cls_range]
						for c in c_cls_range:
							sample_ids = np.where((all_labels == c) == True)[0]
							c_features = all_features[sample_ids]
							c_distribution = sset_distribution_init(self.config, ctl_centroids, c_features)
							ctl_log_prob = np.array([np.sum(ctl_dist.log_prob(torch.tensor(c_features)).numpy(), axis=-1) for ctl_dist in ctl_distributions])
							ctl_score = np.max(ctl_log_prob, axis=0)
							c_log_prob = np.sum(c_distribution.log_prob(torch.tensor(c_features)).numpy(), axis=-1)
							prob_diff = c_log_prob - ctl_score
							threshold = np.quantile(prob_diff, 1-self.config['subset_training']['gmm_ratio'])
							sset_ids = np.where((prob_diff > threshold) == True)[0]
							sset = sample_ids[np.array(sset_ids)]
							ssets += list(sset)
						for c in ctl_cls_range:
							sample_ids = np.where((all_labels == c) == True)[0]
							sset = sample_ids
							ssets += list(sset)
						self.data_loader.train_dataset.adjust_base_indx_tmp(indexs_all[ssets])
						self.data_loader.train_dataset.train_balance_sample()
						print("change train loader")
					elif self.config['trainer']['asym']:
						confusing_pairs = self.config['trainer']['asym_pairs']
						confusing_classes = []
						for p in confusing_pairs:
							cls_a = int(p[0])
							cls_b = int(p[1])
							confusing_classes.append(cls_a)
							confusing_classes.append(cls_b)
							a_sample_ids = np.where((all_labels == cls_a) == True)[0]
							b_sample_ids = np.where((all_labels == cls_b) == True)[0]
							cls_a_features = all_features[a_sample_ids]
							cls_b_features = all_features[b_sample_ids]
							cls_a_naive_distribution = get_distribution(cls_a_features)
							cls_a_centroid = [np.mean(cls_a_features, axis=0)]
							cls_b_naive_distribution = get_distribution(cls_b_features)
							cls_b_centroid = [np.mean(cls_b_features, axis=0)]
							cls_a_cluster_distribution = sset_distribution_init(self.config, cls_b_centroid, cls_a_features)
							cls_b_cluster_distribution = sset_distribution_init(self.config, cls_a_centroid, cls_b_features)
							a_in_a_prob = np.sum(cls_a_cluster_distribution.log_prob(torch.tensor(cls_a_features)).numpy(), axis=-1)
							a_in_b_prob = np.sum(cls_b_cluster_distribution.log_prob(torch.tensor(cls_a_features)).numpy(), axis=-1)
							b_in_b_prob = np.sum(cls_b_cluster_distribution.log_prob(torch.tensor(cls_b_features)).numpy(), axis=-1)
							b_in_a_prob = np.sum(cls_a_cluster_distribution.log_prob(torch.tensor(cls_b_features)).numpy(), axis=-1)
							a_prob_diff = a_in_a_prob - a_in_b_prob
							b_prob_diff = b_in_b_prob - b_in_a_prob
							a_threshold = np.quantile(a_prob_diff, 1-self.config['subset_training']['gmm_ratio'])
							a_sset_ids = np.where((a_prob_diff > a_threshold) == True)[0]
							a_sset = a_sample_ids[np.array(a_sset_ids)]
							ssets += list(a_sset)
							b_threshold = np.quantile(b_prob_diff, 1-self.config['subset_training']['gmm_ratio'])
							b_sset_ids = np.where((b_prob_diff > b_threshold) == True)[0]
							b_sset = b_sample_ids[np.array(b_sset_ids)]
							ssets += list(b_sset)
						left_classes = list(set(np.arange(self.config['num_classes'])) - set(confusing_classes))
						for c in left_classes:
							sample_ids = np.where((all_labels == c) == True)[0]
							sset = sample_ids
							ssets += list(sset)
						self.data_loader.train_dataset.adjust_base_indx_tmp(indexs_all[ssets])
						#self.data_loader.train_dataset.train_balance_sample()
						print("change train loader")

				elif self.config['subset_training']['use_sep_gmm']:
					self.data_loader.train_dataset.switch_data()
					all_labels, all_features, indexs_all = self.get_feature(epoch)
					ssets = []
					ctl_cls_lower_bound = self.config['trainer']['ctl_cls_lower_bound']
					num_ctl = self.config['num_classes']- ctl_cls_lower_bound
					ctl_cls_range = range(ctl_cls_lower_bound, self.config['num_classes'])
					c_cls_range = range(ctl_cls_lower_bound)
					ctl_distributions = [get_distribution(all_features[np.where((all_labels == c) == True)[0]]) for c in ctl_cls_range]
					ctl_centroids = [np.mean(all_features[np.where((all_labels == c) == True)[0]], axis=0) for c in ctl_cls_range]
					for c in c_cls_range:
						sample_ids = np.where((all_labels == c) == True)[0]
						c_features = all_features[sample_ids]
						sset_ids = []
						ctl_log_prob = np.array([np.sum(d.log_prob(torch.tensor(c_features)).numpy(), axis=-1) for d in ctl_distributions])
						gmm_score = np.max(ctl_log_prob, axis=0)
						threshold = np.quantile(gmm_score, self.config['subset_training']['gmm_ratio'])
						sset_ids = np.where((gmm_score < threshold) == True)[0]
						sset = sample_ids[np.array(sset_ids)]
						ssets += list(sset)
					for c in ctl_cls_range:
						sample_ids = np.where((all_labels == c) == True)[0]
						sset = sample_ids
						ssets += list(sset)
					self.data_loader.train_dataset.adjust_base_indx_tmp(indexs_all[ssets])
					#self.data_loader.train_dataset.train_balance_sample()
					print("change train loader")
				
				result= self._train_epoch(epoch)

			# save logged informations into log dict
			log = {'epoch': epoch}
			for key, value in result.items():
				if key == 'metrics':
					log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
				elif key == 'metrics_gt':
					log.update({mtr.__name__ + '_gt': value[i] for i, mtr in enumerate(self.metrics)})
				elif key == 'val_metrics':
					log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
				elif key == 'test_metrics':
					log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
				else:
					log[key] = value
			
			try:
				wandb.log(log)
			except:
				print("wandb not initialized")
			
			# print logged informations to the screen
			for key, value in log.items():
				self.logger.info('    {:15s}: {}'.format(str(key), value))

			# evaluate model performance according to configured metric, save best checkpoint as model_best
			best = False
			if self.mnt_mode != 'off':
				try:
					# check whether model performance improved or not, according to specified metric(mnt_metric)
					improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
							   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
				except KeyError:
					self.logger.warning("Warning: Metric '{}' is not found. "
										"Model performance monitoring is disabled.".format(self.mnt_metric))
					self.mnt_mode = 'off'
					improved = False

				if improved:
					self.mnt_best = log[self.mnt_metric]
					not_improved_count = 0
					best = True
				else:
					not_improved_count += 1

				if not_improved_count > self.early_stop:
					self.logger.info("Validation performance didn\'t improve for {} epochs. "
									 "Training stops.".format(self.early_stop))
					break

			if epoch % self.save_period == 0:
				self._save_checkpoint(epoch, save_best=best)
		
	
	def _prepare_device(self, n_gpu_use):
		"""
		setup GPU device if available, move model into configured device
		"""
		n_gpu = torch.cuda.device_count()
		if n_gpu_use > 0 and n_gpu == 0:
			self.logger.warning("Warning: There\'s no GPU available on this machine,"
								"training will be performed on CPU.")
			n_gpu_use = 0
		if n_gpu_use > n_gpu:
			self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
								"on this machine.".format(n_gpu_use, n_gpu))
			n_gpu_use = n_gpu
		device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
		list_ids = list(range(n_gpu_use))
		return device, list_ids

	def _save_checkpoint(self, epoch, save_best=False):
		"""
		Saving checkpoints

		:param epoch: current epoch number
		:param log: logging information of the epoch
		:param save_best: if True, rename the saved checkpoint to 'model_best.pth'
		"""
		arch = type(self.model).__name__

		state = {
			'arch': arch,
			'epoch': epoch,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'monitor_best': self.mnt_best
		}
		# filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
		# torch.save(state, filename)
		# self.logger.info("Saving checkpoint: {} ...".format(filename))
		if save_best:
			if self.parse.distillation:
				model_name = self.parse.distill_mode + '_' + 'model_best' + str(self.parse.dataseed) + '.pth'
				if not self.parse.reinit:
					model_name = 'keep_' + model_name
			else:
				model_name = 'model_best' + str(self.parse.dataseed) + '.pth'
			
			
			
			best_path = str(self.checkpoint_dir / model_name)
			torch.save(state, best_path)
			self.logger.info("Saving current best: " + model_name + " at: {} ...".format(best_path))


	def _resume_checkpoint(self, resume_path):
		"""
		Resume from saved checkpoints

		:param resume_path: Checkpoint path to be resumed
		"""
		resume_path = str(resume_path)
		self.logger.info("Loading checkpoint: {} ...".format(resume_path))
		checkpoint = torch.load(resume_path)
		self.start_epoch = checkpoint['epoch'] + 1
		self.mnt_best = checkpoint['monitor_best']

		# load architecture params from checkpoint.
		if checkpoint['config']['arch'] != self.config['arch']:
			self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
								"checkpoint. This may yield an exception while state_dict is being loaded.")
		self.model.load_state_dict(checkpoint['state_dict'])

		# load optimizer state from checkpoint only when optimizer type is not changed.
		if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
			self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
								"Optimizer parameters not being resumed.")
		else:
			self.optimizer.load_state_dict(checkpoint['optimizer'])

		self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


