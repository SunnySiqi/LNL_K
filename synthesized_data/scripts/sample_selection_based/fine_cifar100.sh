#!/bin/bash

python main.py -d 0 -c hyperparams/multistep/config_cifar100_cce_rn34.json --lr_scheduler multistep --arch rn34 --loss_fn cce --dataset cifar10 --traintools robustloss --no_wandb --dynamic --distill_mode fine-gmm --seed 456 --warmup 40
