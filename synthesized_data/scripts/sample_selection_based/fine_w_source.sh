#!/bin/bash

python main.py -d 0 -c hyperparams/multistep/config_cifar10_fine_w_source_rn34.json --lr_scheduler multistep --arch rn34  --loss_fn cce --traintools robustloss --no_wandb --seed 123 --warmup 40 --dynamic --distill_mode fine-gmm --every 10
