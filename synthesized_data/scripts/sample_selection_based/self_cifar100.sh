#!/bin/bash

python main.py -d 0  -c hyperparams/multistep/config_cifar100_self_rn34.json --lr_scheduler multistep --arch rn34  --traintools robustloss --no_wandb --seed 456 --warmup 40
