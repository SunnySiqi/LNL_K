#!/bin/bash

python main.py -d 1 -c hyperparams/multistep/config_cifar10_fine_w_source_rn34.json --lr_scheduler multistep --arch rn34 --loss_fn cce --dataset cifar10 --traintools robustloss --no_wandb --dynamic --distill_mode fine-gmm --seed 456 --warmup 40
