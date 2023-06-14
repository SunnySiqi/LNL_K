#!/bin/bash

python main.py -d 1  -c hyperparams/multistep/config_cifar10_cluster_rn34.json --lr_scheduler multistep --arch rn34  --traintools robustloss --no_wandb --seed 123 --warmup 40
