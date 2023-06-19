from __future__ import print_function
import argparse
import os
import sys
# adding adaptation_methods to the system path
sys.path.insert(0, '/net/ivcfs5/mnt/data/swang/research/LNL+K/LNL_K/adaptation_methods')
import shutil
import json
import numpy as np
import torch, sklearn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import grad
from tqdm import tqdm
from simple_multi_dataloader import TripletImageLoader, PLDataModule
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from adaptation_methods import *


# Training settings
parser = argparse.ArgumentParser(description='Drug Discovery')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--predict-batch-size', type=int, default=128,
                    help='input batch size for prediction (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=float, default=0.01, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Drug_Discovery', type=str,
                    help='name of experiment')
parser.add_argument('--datadir', default='/projectnb/morphem/data', type=str,
                    help='directory of the single cell dataset (default: data)')
parser.add_argument('--dataset', default='taorf|cdrp|bbbc022', type=str, choices=['cdrp', 'bbbc022', 'cdrp|bbbc022', 'taorf', 'taorf|cdrp', 'taorf|bbbc022', 'taorf|cdrp|bbbc022'],
                    help='dataset for training and validation')
parser.add_argument('--eval_dataset', default='taorf|cdrp|bbbc022', type=str, choices=['cdrp', 'bbbc022', 'cdrp|bbbc022', 'taorf', 'taorf|cdrp', 'taorf|bbbc022', 'taorf|cdrp|bbbc022'],
                    help='dataset for testing')
parser.add_argument('--domain_label', default='dataset', type=str, choices=['dataset', 'Metadata_Well', 'Metadata_Plate', 'Metadata_Site'],
                    help='domain_label for the single cell images')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--test_weak', dest='test_weak', action='store_true', default=False,
                    help='To only run inference on test weak reaction set')
parser.add_argument('--test_normal', dest='test_normal', action='store_true', default=False,
                    help='To only run inference on test normal reaction set')
parser.add_argument('--test_strong', dest='test_strong', action='store_true', default=False,
                    help='To only run inference on test strong reaction set')
parser.add_argument('--test_class_num', type=int, default=100, metavar='test_cls_num',
                    help='How many classes are there in the test set (default: 100)')
parser.add_argument('--dim_embed', type=int, default=256, metavar='N',
                    help='how many dimensions in embedding (default: 256)')
parser.add_argument('--margin', type=float, default=0.3, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--augmentation_method', type=str, default='none', choices=['none', 'cutmix', 'mixup'],
                    help='specifies what kind of domain adaptation method to use, if any')
parser.add_argument('--net', type=str, default='full_effb0', choices=['res18', 'res50', 'effb0', 'full_effb0'],
                    help='underlying network backbone to use')
parser.add_argument('--split', type=str, default='Cells_Out', choices=['Cells_Out', 'Replicates_Out'],
                    help='random split identifier')
parser.add_argument('--controls', action='store_true', default=False,
                    help='test split separates mechanstic classes from controls')
parser.add_argument('--treatment', action='store_true', default=False,
                    help='test split separates treatments that share a mechanstic class from others')
parser.add_argument('--num_processors', type=int, default=4)
parser.add_argument('--num_gpus', type=int, default=4)
parser.add_argument('--max_images_per_treatment', type=int, default=3500, metavar='M',
                    help='number of images to use from each dataset during training, M <= 0 indicates using all the images')
parser.add_argument('--max_num_treatment', type=int, default=100, metavar='M',
                    help='number of treatments, M <= 0 indicates using all the images')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--whiteningnorm', action='store_true', default=False)
parser.add_argument('--reg_param', type=float, default=1e-2)
parser.add_argument('--r', type=float, default=2.0,
                    help="Distance threshold (i.e. radius) in calculating clusters.")
parser.add_argument('--use_crust', action='store_true',
                    help="Whether to use crust in dataset.")
parser.add_argument('--use_crust_k', action='store_true',
                    help="Whether to use crust with control knowledge in dataset.")
parser.add_argument('--num_treatment_class', type=int, default=1)
parser.add_argument('--fl-ratio', type=float, default=0.8,
                    help="Ratio for number of facilities.")
parser.add_argument('--use_fine', action='store_true',
                    help="select coresets with FINE")
parser.add_argument('--use_fine_k', action='store_true',
                    help="select coresets with FINE_w")
parser.add_argument('--use_self_filter', action='store_true',
                    help="select coresets with self-filter method")
parser.add_argument('--use_self_filter_k', action='store_true',
                    help="select coresets with self_filter_w")
parser.add_argument('--self_filter_k', type=int, default=3, metavar='self_filter_k',
                    help='The size of the self-filter memory bank')
parser.add_argument('--treatment_hard', action='store_true',
                    help="use the hard treatments mixture")

def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print(args)
    test_loader = torch.utils.data.DataLoader(TripletImageLoader(args, 'test', args.eval_dataset),batch_size=args.batch_size, shuffle=False, num_workers=args.num_processors, pin_memory=True)
    if args.test:
        logger = TensorBoardLogger("tb_logs", name='test_'+args.name)
        checkpoint = torch.load(args.resume)
        best_model = LitCNN(args, args.test_class_num)
        best_model.load_state_dict(checkpoint)
        trainer = pl.Trainer(devices=args.num_gpus, accelerator="gpu", strategy="dp", logger=logger)
        best_model.eval()
        trainer.test(best_model, test_loader)
    else:
        best_acc = 0
        # optionally resume from a checkpoint
        cudnn.benchmark = True

        train_data_module = PLDataModule(args)
        train_loader = train_data_module.train_dataloader()
        val_loader = train_data_module.val_dataloader()
        nb_classes = len(list(train_loader.dataset.treatment_count_id.keys()))
        if args.use_crust:
            store_name = '_'.join([args.dataset, 'use_crust', 'fl_ratio('+str(args.fl_ratio)+')', 'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = crust_LitCNN(args, nb_classes, train_data_module)
        elif args.use_crust_k:
            store_name = '_'.join([args.dataset, 'use_crust_k', 'fl_ratio('+str(args.fl_ratio)+')', 'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = crust_LitCNN(args, nb_classes, train_data_module)        
        elif args.use_fine:
            store_name = '_'.join([args.dataset, 'use_fine',  'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = fine_LitCNN(args, nb_classes, train_data_module)
        elif args.use_fine_k:
            store_name = '_'.join([args.dataset, 'use_fine_k', 'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = fine_LitCNN(args, nb_classes, train_data_module)
        elif args.use_self_filter:
            store_name = '_'.join([args.dataset, 'use_sft',  'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = sft_LitCNN(args, nb_classes, train_data_module)
        elif args.use_self_filter_k:
            store_name = '_'.join([args.dataset, 'use_sft_k', 'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = sft_LitCNN(args, nb_classes, train_data_module)
        else:
            store_name = '_'.join([args.dataset, 'bs('+str(args.batch_size)+')', 'lr('+str(args.lr)+')', str(args.max_images_per_treatment), str(args.max_num_treatment)])
            cnn_model = LitCNN(args, nb_classes, train_data_module)
        # tb_filename = os.path.join('tb_logs', store_name)
        # if not os.path.exists(tb_filename):
        #     os.makedirs(tb_filename)
        logger = TensorBoardLogger("tb_logs", name=store_name)
        checkpoint_callback = ModelCheckpoint(monitor='validation_top1acc', mode='max', dirpath='saved/',filename=args.name+'_number_class_'+str(nb_classes)+'_{epoch:02d}-{val_loss:.2f}')
        trainer = pl.Trainer(max_epochs=args.epochs, devices=args.num_gpus, accelerator="gpu", strategy="dp", logger=logger, log_every_n_steps=10, reload_dataloaders_every_n_epochs=1, callbacks=[checkpoint_callback])
        trainer.fit(cnn_model, train_loader, val_loader)
        #tf_writer = SummaryWriter(log_dir=os.path.join('runs', store_name))

        checkpoint = torch.load(checkpoint_callback.best_model_path)
        best_model = LitCNN(args, nb_classes, train_data_module)
        best_model.load_state_dict(checkpoint['state_dict'])
        trainer = pl.Trainer(devices=args.num_gpus, accelerator="gpu", strategy="dp", logger=logger)
        trainer.test(best_model, test_loader)
        # trainer.test(val_loader, ckpt_path="best")



# mixup: somewhere in main_moco.py
def mixup(input, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample([input.shape[0]]).to(device=input.device)
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam

# cutmix: somewhere in main_moco.py
def cutmix(input, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample().to(device=input.device)
    lam = torch.max(lam, 1. - lam)
    (bbx1, bby1, bbx2, bby2), lam = rand_bbox(input.shape[-2:], lam)
    output = input.clone()
    output[..., bbx1:bbx2, bby1:bby2] = output[randind][..., bbx1:bbx2, bby1:bby2]
    return output, randind, lam

def rand_bbox(size, lam):
    W, H = size
    cut_rat = (1. - lam).sqrt()
    cut_w = (W * cut_rat).to(torch.long)
    cut_h = (H * cut_rat).to(torch.long)

    cx = torch.zeros_like(cut_w, dtype=cut_w.dtype).random_(0, W)
    cy = torch.zeros_like(cut_h, dtype=cut_h.dtype).random_(0, H)

    bbx1 = (cx - cut_w // 2).clamp(0, W)
    bby1 = (cy - cut_h // 2).clamp(0, H)
    bbx2 = (cx + cut_w // 2).clamp(0, W)
    bby2 = (cy + cut_h // 2).clamp(0, H)

    new_lam = 1. - (bbx2 - bbx1).to(lam.dtype) * (bby2 - bby1).to(lam.dtype) / (W * H)

    return (bbx1, bby1, bbx2, bby2), new_lam

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



if __name__ == '__main__':
    main()    
