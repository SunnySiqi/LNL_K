from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import json
import torch
import pickle
import h5py
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.autograd import Variable
from tqdm import tqdm
from skimage.util import random_noise
import pandas as pd
import random
import torch.nn as nn
from normalizer import WhiteningNormalizer
import torchvision.models as models
import pytorch_lightning as pl

def default_image_loader(fn):#, means, stds):
	im = plt.imread(fn)#.transpose(2, 0, 1)
	im = np.reshape(im, (im.shape[0], im.shape[0], -1), order="F").transpose(2, 0, 1)
	#image input shape (160, 960) ,(h, w*c) -> reshape (h, w, c) = (160, 160, 6). CNN takes 5 channel input
	im = im[:5,:,:]
	# PIL_image = Image.fromarray(im.astype('uint8'), 'RGB')
	return im
	#return torch.from_numpy(im)

def gaussian_noise_image_loader(fn):
	im = plt.imread(fn)
	im = random_noise(im, mode='gaussian', seed=None, clip=True)
	im = np.reshape(im, (im.shape[0], im.shape[0], -1), order="F").transpose(2, 0, 1)
	return torch.Tensor(im)

def parse_data(args, dataset_list):
	for dataset in dataset_list:
		train, test, val = [], [], []
		treatment_list = set()
		if dataset == 'bbbc022':
			metadata_filename = 'sc-metadata-' + dataset + '.csv'
		else:
			metadata_filename = dataset.upper() + '-sc-metadata.csv'

		metadata_filename = os.path.join(args.datadir, 'metadata', dataset.upper(), metadata_filename)
		dataset_metadata = pd.read_csv(metadata_filename)
		for _, item in tqdm(dataset_metadata.iterrows(), desc=('loading dataset ' + dataset), total=dataset_metadata.shape[0]):
			split_id = item[args.split]
			if args.domain_label == 'dataset':
				domain = dataset
			else:
				domain = item[args.domain_label]
			well = item['Metadata_Well']
			plate = item['Metadata_Plate']
			site = item['Metadata_Site']

			treatment = item['Treatment']
			if treatment in ['DMSO@0', 'NA@NA', 'EMPTY_']:
				treatment = 'control'
			treatment_list.add(treatment)

			imname = item['Image_Name']
			if split_id == 0:
				train.append([imname, treatment, dataset, well, plate, site, domain])
			elif split_id == 1:
				val.append([imname, treatment, dataset, well, plate, site, domain])
			elif split_id == 2:
				test.append([imname, treatment, dataset, well, plate, site, domain])
			else:
				assert False


		for split, imnames in zip(['train', 'test', 'val'], [train, test, val]):
			filename = os.path.join('data', dataset.lower() + '_' + split + '_' + args.domain_label + '_' + args.split + '.pkl')
			pickle.dump({'data' : imnames}, open(filename, 'wb'))
		treatment_list = list(treatment_list)
		treatment2index = dict(zip(treatment_list, range(len(treatment_list))))
		filename = os.path.join('data', dataset.lower() + '_' + 'treatment2id.pkl')
		if not os.path.exists(filename):
			pickle.dump({'data' : treatment2index}, open(filename, 'wb'))


class TripletImageLoader(torch.utils.data.Dataset):
	def __init__(self, args, split, dataset_list, transform=None, loader=default_image_loader):
		dataset_list = dataset_list.split('|')
		self.args = args
		self.impath = os.path.join(args.datadir, 'images')
		self.transform = transform
		self.loader = loader
		if split == 'test':
			self.dataset = dataset_list[0]
		else:
			self.dataset = dataset_list
		
		self.split = split
		self.max_img_t = -1
		if args.max_images_per_treatment != -1:
			if self.split == 'train':
				self.max_img_t = args.max_images_per_treatment
			elif self.split == 'val':
				# self.max_img_t = 0.1*args.max_images_per_treatment
				self.max_img_t = -1
		
		self.dataset2controlid = {'bbbc022' : 'DMSO@0', 'cdrp' : 'NA@NA', 'taorf' : 'EMPTY_'}
		self.imnames = []
		self.resize = transforms.Resize(size=256)
		self.crop = transforms.CenterCrop(size=224)
		for dataset in dataset_list:
			filename = os.path.join('data', dataset.lower() + '_' + split + '_' + args.domain_label + '_' + args.split + '.pkl')
			if not os.path.exists(filename):
				parse_data(args, dataset_list)
			treatment2id_filename = os.path.join('data', dataset.lower() + '_' + 'treatment2id.pkl')
			self.treatment2index = pickle.load(open(treatment2id_filename, 'rb'))['data']
			self.imnames += pickle.load(open(filename, 'rb'))['data']

		if self.args.treatment_hard:
			select_treatment_file_name = os.path.join('data', dataset.lower() + '_' + 'select_hard_t_id.pkl')
		else:
			select_treatment_file_name = os.path.join('data', dataset.lower() + '_' + 'select_t_id.pkl')
		select_treatment_list_id = pickle.load(open(select_treatment_file_name, 'rb'))['data']
		self.new_treat2id_dict = dict(zip(select_treatment_list_id, range(len(select_treatment_list_id))))
		if args.test_strong:
			select_treatment_list_id = select_treatment_list_id[1:36]
		elif args.test_normal:
			select_treatment_list_id = select_treatment_list_id[36:86]
		elif args.test_weak:
			select_treatment_list_id = select_treatment_list_id[0] + select_treatment_list_id[86:]
		treatment_list, domain_list = set(), set()
		self.treatment2id = {}
		imnames = []
		self.treatment_count = {}
		self.treatment_count_id = {}
		if not args.test_normal or not args.test_strong:
			self.treatment_count['control'] = 0
			self.treatment_count_id['control'] = []
		for i, (_, treatment, _, well, plate, site, domain) in enumerate(self.imnames):
			if treatment in ['DMSO@0', 'NA@NA', 'EMPTY_']:
				treatment = 'control'
				if args.test_normal or args.test_strong or self.treatment_count[treatment] > self.max_img_t and self.max_img_t != -1:
					continue
			else:
				treat_label = treatment
				if treat_label not in self.treatment2index:
					continue
				if self.treatment2index[treat_label] not in select_treatment_list_id:
					continue
				if treat_label not in self.treatment_count:
					self.treatment_count[treat_label] = 0
					self.treatment_count_id[treat_label] = []
				self.treatment_count[treat_label] += 1
				if self.treatment_count[treat_label] > self.max_img_t and self.max_img_t != -1:
					continue
			imnames.append(self.imnames[i])
			treatment_list.add(treatment)
			domain_list.add(domain)


			if treatment not in self.treatment2id:
				self.treatment2id[treatment] = {}
			if plate not in self.treatment2id[treatment]:
				self.treatment2id[treatment][plate] = {}
			if well not in self.treatment2id[treatment][plate]:
				self.treatment2id[treatment][plate][well] = {}
			if site not in self.treatment2id[treatment][plate][well]:
				self.treatment2id[treatment][plate][well][site] = set()

			self.treatment2id[treatment][plate][well][site].add(len(imnames) - 1)
			self.treatment_count_id[treat_label].append(len(imnames) - 1)

		self.imnames = imnames
		self.treatment2moa = {}
		if split != 'train':
			for dataset in dataset_list:
				moa_label = 'Metadata_moa.x'
				treatment_id = 'Var1'
				if dataset == 'taorf':
					moa_label = moa_label.split('.')[0]
					treatment_id = 'pert_name'

				filename = os.path.join(args.datadir, 'metadata', dataset.upper(), dataset.upper() + '_MOA_MATCHES_official.csv')
				moa_file = pd.read_csv(filename)
				for _, item in moa_file.iterrows():
					moa_treatment = item[treatment_id]
					if moa_treatment not in self.treatment2id:
						continue

					if moa_treatment != self.dataset2controlid[dataset]:
						self.treatment2moa[moa_treatment] = set(item[moa_label].split('|'))

		self.query_treatment = self.get_test_query_treatment()
		treatment_list, domain_list = list(treatment_list), list(domain_list)
		#self.treatment2index = dict(zip(treatment_list, range(len(treatment_list))))
		# treatment2id_filename = os.path.join('data', dataset.lower() + '_' + 'treatment2id.pkl')
		# self.treatment2index = pickle.load(open(treatment2id_filename, 'rb'))['data']
		self.domain2index = dict(zip(domain_list, range(len(domain_list))))
		self.train_imnames = self.imnames.copy()
		self.median_count = np.median(np.array(list(self.treatment_count.values())))
		# if split == 'val' or split == 'test':
		# 	self.train_balance_sample()

	def train_balance_sample(self):
		sset = []
		if self.max_img_t == -1:
			sample_num = self.median_count
		else:
			sample_num = np.minimum(self.median_count, self.max_img_t)
		print("SAMPLE NUM!!!", sample_num)
		print("TREATMENT NUM!!!", len(list(self.treatment_count.keys())))
		# print("TREATMENT NUM!!!", len(list(self.treatment_count_id.keys())))
		for t in self.treatment_count_id.keys():
			sub_t = np.random.choice(np.array(self.treatment_count_id[t]), int(sample_num))
			sset.append(sub_t)
		sset = np.concatenate(sset, axis=0)
		self.imnames = np.array(self.imnames)
		new_data = self.imnames[sset].copy()
		self.train_imnames = new_data
		print("!!!!!!!BALANCE!!!!!",str(self.split)+":"+str(len(self.train_imnames)))

	def get_test_query_treatment(self):
		moa2treatment = {}
		for t in self.treatment2moa:
			for moa in self.treatment2moa[t]:
				if moa not in moa2treatment:
					moa2treatment[moa] = [t]
				else:
					moa2treatment[moa].append(t)
		query_moa = [moa for moa in list(moa2treatment.keys()) if len(moa2treatment[moa]) > 1]
		query_treatment = [t for t in list(self.treatment2moa.keys()) if len(self.treatment2moa[t].intersection(query_moa)) > 0]
		return query_treatment

	def get_treatment_emb(self, embeds):
		treatment_well_feature = {}
		for t in self.treatment2id:
			treatment_well_feature[t] = []
			for p in self.treatment2id[t]:
				for w in self.treatment2id[t][p]:
					site_features = {}
					for s in self.treatment2id[t][p][w]:
						cell_ids = self.treatment2id[t][p][w][s]
						site_embeds = embeds[list(cell_ids)]
						site_features[s] = np.median(site_embeds, axis=0)
					well_embeds = np.array(list(site_features.values()))
					treatment_well_feature[t].append(np.mean(well_embeds, axis=0))
		
		#load well level features from Nikita's pretrained
		#treatment_well_feature = pickle.load(open('treatment_well.pkl', 'rb'))

		if self.args.whiteningnorm:
			controls = np.array(treatment_well_feature['control'])
			whN = WhiteningNormalizer(controls, self.args.reg_param)
			wn_treatment_well_feature = {}
			for t in treatment_well_feature:
				wn_treatment_well_feature[t] = []
				for f in treatment_well_feature[t]:
					wn_treatment_well_feature[t].append(whN.normalize(f))
			for key in wn_treatment_well_feature:
				wn_treatment_well_feature[key] = np.mean(np.array(wn_treatment_well_feature[key]),axis=0)
			return wn_treatment_well_feature
		else:
			for key in treatment_well_feature:
				treatment_well_feature[key] = np.mean(np.array(treatment_well_feature[key]),axis=0)
			return treatment_well_feature

	def test_treatment(self, embeds):
		""" Returns the accuracy of the fill in the blank task
			embeds: precomputed embedding features used to score
					each compatibility question
		"""

		firstHit = np.array([])
		preAtK = np.array([])
		cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		embeds = embeds.data.cpu().numpy()
		treatment_emb = self.get_treatment_emb(embeds)
		for i, treatment in enumerate(self.query_treatment):
			i_emb = torch.Tensor(treatment_emb[treatment])
			treatment_list = list(self.treatment2id.keys())
			treatment_list = [t for t in treatment_list if t != treatment and t != 'control']
			treatment_sim = torch.zeros(len(treatment_list))
			for j, paired_treatments in enumerate(treatment_list):
				j_emb = torch.Tensor(treatment_emb[paired_treatments])
				treatment_sim[j] = cos(i_emb, j_emb)

			_, order = treatment_sim.topk(len(treatment_sim))
			num_samples = round(len(order)*self.args.test_k)
			num_success = 0
			count_samples = 0
			for j, next_id in enumerate(order):
				treat_label = treatment_list[next_id].split('@')[0]
				if treat_label not in self.treatment2moa:
					continue
				next_moa = self.treatment2moa[treat_label]
				count_samples += 1
				is_match = len(self.treatment2moa[treatment].intersection(next_moa)) > 0
				if is_match:
					if count_samples > num_samples and num_success == 0:
						firstHit = np.append(firstHit, count_samples)
						break
					elif count_samples <= num_samples:
						if num_success == 0:
							firstHit = np.append(firstHit, count_samples)
						num_success += 1
					elif count_samples > num_samples and num_success > 0:
						break
			preAtK = np.append(preAtK, float(num_success)/num_samples)

		#recallAtK = (firstHit <= num_samples).astype(np.float).mean()
		preAtK = np.mean(preAtK)
		medianR = np.median(firstHit)
		return preAtK, medianR, len(firstHit)

	def random_illumination(self, image):
		# Make channels independent images
		numchn = 5
		source = torch.unsqueeze(image, 1)
		gray_to_rgb = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
		source = gray_to_rgb(source)
	
		# Apply illumination augmentations
		bright = torch.from_numpy(np.random.uniform(low=0.8, high=1.2, size=numchn))
		channels = [transforms.functional.adjust_brightness(source[s,...], bright[s]) for s in range(numchn)]
		contrast = torch.from_numpy(np.random.uniform(low=0.8, high=1.2, size=numchn))
		channels = [transforms.functional.adjust_contrast(channels[s], contrast[s]) for s in range(numchn)]
		result = torch.cat([torch.unsqueeze(t, 0) for t in channels], axis=0)
		
		# Recover multi-channel image
		result = result[:, 0, :, :]
		result = torch.squeeze(result)
		result = [(result[i, :, :]-result[i, :, :].min())/(result[i, :, :].max()-result[i, :, :].min()) for i in range(numchn)]
		result = torch.cat([torch.unsqueeze(t, 0) for t in result], axis=0)
		return result

	def load_img(self, img_idx):
		imname, _, dataset, _, _, _, _ = self.train_imnames[img_idx]
		img = self.loader(os.path.join(self.impath, dataset.upper(), imname))
		img = self.resize(torch.Tensor(img))
		if self.split == 'train':
			img = self.crop(img)
			img = self.random_illumination(img)
		# img_min = img.min()
		# img_max = img.max()
		# norm_img = (img-img_min)/(img_max-img_min)*2.0 - 1.0
		#norm_img = torch.stack([torch.Tensor(self.transform(Image.fromarray(img[i].astype('uint8'), 'RGB')))for i in range(5)])
		if self.transform is not None:
			img = self.transform(norm_img)
			#img = torch.stack([torch.Tensor(self.transform(Image.fromarray(img[i].astype('uint8'), 'RGB')))for i in range(num_channels)])
		return img

	def get_pos(self, treatment):
		pos_candidates = set()
		for p in self.treatment2id[treatment]:
			for w in self.treatment2id[treatment][p]:
				for s in self.treatment2id[treatment][p][w]:
					pos_candidates = pos_candidates.union(self.treatment2id[treatment][p][w][s])
		pos = random.sample(pos_candidates, 1)[0]
		return pos
	
	def switch_data(self):
		self.train_balance_sample()
		#self.train_imnames = self.imnames.copy()
		print("!!!!!!!!!!!!SWITCHDATA!!!", len(self.train_imnames))
	
	def adjust_base_indx(self, idx):
		self.train_imnames = np.array(self.train_imnames)
		print("!!!!!!!!!!!!ADJUST INDEX!!!", len(self.train_imnames))
		new_data = self.train_imnames[idx].copy()
		self.train_imnames = new_data
		#print("!!!!!!!!!!!!ADJUSTBASEINDEX", len(new_data))

	def __getitem__(self, index):
		img = self.load_img(index)
		_, treatment, _, _, _, _,domain = self.train_imnames[index]
		if treatment in ['DMSO@0', 'NA@NA', 'EMPTY_']:
			treatment = 'control'
		treatment_id = self.new_treat2id_dict[self.treatment2index[treatment]]
		return img, treatment_id, index

	def __len__(self):
		return len(self.train_imnames)

class PLDataModule(pl.LightningDataModule):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.train_dataset = TripletImageLoader(self.args, 'train', self.args.dataset)
		self.train_dataset.train_balance_sample()

	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=self.args.batch_size,
			shuffle=True,
			num_workers=self.args.num_processors,
			pin_memory=True
		)
	def predict_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=self.args.predict_batch_size,
			shuffle=False,
			num_workers=self.args.num_processors,
			pin_memory=True
		)
	def val_dataloader(self):
		val_dataset = TripletImageLoader(self.args, 'val', self.args.dataset)
		print("VAL DATASET!!!!",len(val_dataset.train_imnames))
		#val_dataset.train_balance_sample()
		return torch.utils.data.DataLoader(
			val_dataset,
			batch_size=self.args.batch_size,
			shuffle=False,
			num_workers=self.args.num_processors,
			pin_memory=True
		)

