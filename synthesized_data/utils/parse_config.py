import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from logger.logger import setup_logging
# from logger import setup_logging
from .util import read_json, write_json

class ConfigParser:

	__instance = None

	def __new__(cls, args, options='', timestamp=True):
		raise NotImplementedError('Cannot initialize via Constructor')

	@classmethod
	def __internal_new__(cls):
		return super().__new__(cls)

	@classmethod
	def get_instance(cls, args=None, options='', timestamp=True):
		if not cls.__instance:
			if args is None:
				NotImplementedError('Cannot initialize without args')
			cls.__instance = cls.__internal_new__()
			cls.__instance.__init__(args, options)

		return cls.__instance

	def __init__(self, args, options='', timestamp=True):
		# parse default and custom cli options
		for opt in options:
			args.add_argument(*opt.flags, default=None, type=opt.type)
		args = args.parse_args()
		self.args = args
		
		# set config file from arguments (dataset, lr scheduler, loss fn)
		cfg_fname=None
		if args.config is not None:
			cfg_fname = args.config
		elif args.dataset and args.lr_scheduler and args.loss_fn:
			cfg_fname = './hyperparams/' + args.lr_scheduler + '/config_' + args.dataset + '_' + args.loss_fn + '_' + args.arch + '.json'
		
		if args.device:
			os.environ["CUDA_VISIBLE_DEVICES"] = args.device
		if args.resume is None:
			msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
			if cfg_fname is not None:
				self.cfg_fname = Path(cfg_fname)
			else:
				assert args.config is not None, msg_no_cfg
				self.cfg_fname = Path(args.config)
			config = read_json(self.cfg_fname)
			self.resume = None
		else:
			self.resume = Path(args.resume)
			resume_cfg_fname = self.resume.parent / 'config_123.json'
			config = read_json(resume_cfg_fname)
			if args.config is not None:
				config.update(read_json(Path(args.config)))

		# load config file and apply custom cli options
		self._config = _update_config(config, options, args)

		# set save_dir where trained model and log will be saved.
		save_dir = Path(self.config['trainer']['save_dir'])

		dataset_name = self.config['name'].split('_')[0]
		model_type = self.config['arch']['type']
		lr_scheduler = self.config['lr_scheduler']['type']
		loss_fn = self.config['train_loss']['type']
		sym_setting = 'control' if not self.config['trainer']['asym'] else 'asym'
		percent = str(int(self.config['trainer']['percent']*100))
		if self.config['subset_training']['use_crust']:
			method = 'crust'
		elif self.config['subset_training']['adptive_crust']:
			method = 'adptive_crust'
		elif self.config['subset_training']['oracle']:
			method = 'oracle'
		elif self.config['subset_training']['use_class_gmm']:
			if self.config['subset_training']['naive_init']:
				method = 'naive_cluster'
			elif self.config['subset_training']['naive_centroid_init']:
				method = 'naive_centroid_init'
			else:
				method = 'cluster'
		elif self.config['train_loss']['type'] == 'CoteachingPlusLoss' or self.config['train_loss']['type'] == 'CoteachingLoss':
			method = 'co-teaching'
		elif self.config['subset_training']['self_filter']:
			method = 'self_filter'
		elif self.config['subset_training']['self_filter_w']:
			method = 'self_filter_w'
		elif self.config['subset_training']['fine_with_source']:
			if self.config['subset_training']['clean_epoch'] > 0:
				method = 'clean_FINE_w'
			else:
				method = 'FINE_w_noise_source'
				print("METHOD: FINE_w_noise_source!!!!!!!!!!!!!!")
		elif self.config['train_loss']['type'] == 'CrossEntropyLoss':
			method = 'standard'
		else:
			method = 'FINE'
			print("METHOD: FINE!!!!!!!!!!!!!")

		if args.resume is None:
			if args.distillation:
				distill_mode = args.distill_mode
				seed = args.dataseed
				self._save_dir = save_dir / 'models' / dataset_name / model_type / lr_scheduler / loss_fn / sym_setting / percent / distill_mode / str(int(seed))
				self._log_dir = save_dir / 'log' / dataset_name / model_type / loss_fn / sym_setting / percent / distill_mode / str(int(seed))
			else:
				self._save_dir = save_dir / 'models' / dataset_name / model_type / lr_scheduler / loss_fn / sym_setting / method / percent
				self._log_dir = save_dir / 'log' / dataset_name / model_type /  loss_fn / sym_setting / method / percent
			print("LOG DIR!!!!!!!!!!!!", self._log_dir)
			self.save_dir.mkdir(parents=True, exist_ok=True)
			config_name = 'config_' + str(self.config['seed']) + '.json'
			write_json(self.config, self.save_dir / config_name)
		else:
			self._log_dir = save_dir / 'inf_log' / dataset_name / model_type /  loss_fn / sym_setting / method / percent
			
		self.log_dir.mkdir(parents=True, exist_ok=True)

		# configure logging module
		setup_logging(self.log_dir)
		self.log_levels = {
			0: logging.WARNING,
			1: logging.INFO,
			2: logging.DEBUG
		}

	def initialize(self, name, module, *args, **kwargs):
		"""
		finds a function handle with the name given as 'type' in config, and returns the 
		instance initialized with corresponding keyword args given as 'args'.
		"""
		module_name = self[name]['type']
		module_args = dict(self[name]['args'])
		assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
		module_args.update(kwargs)
		return getattr(module, module_name)(*args, **module_args)

	def __getitem__(self, name):
		return self.config[name]

	def get_logger(self, name, verbosity=2):
		msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
																					   self.log_levels.keys())
		assert verbosity in self.log_levels, msg_verbosity
		logger = logging.getLogger(name)
		logger.setLevel(self.log_levels[verbosity])
		return logger

	# setting read-only attributes
	@property
	def config(self):
		return self._config

	@property
	def save_dir(self):
		return self._save_dir

	@property
	def log_dir(self):
		return self._log_dir


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
	for opt in options:
		value = getattr(args, _get_opt_name(opt.flags))
		if value is not None:
			_set_by_path(config, opt.target, value)
			if 'target2' in opt._fields:
				_set_by_path(config, opt.target2, value)
			if 'target3' in opt._fields:
				_set_by_path(config, opt.target3, value)

	return config


def _get_opt_name(flags):
	for flg in flags:
		if flg.startswith('--'):
			return flg.replace('--', '')
	return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
	"""Set a value in a nested object in tree by sequence of keys."""
	_get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
	"""Access a nested object in tree by sequence of keys."""
	return reduce(getitem, keys, tree)
