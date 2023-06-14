#!/bin/bash

python main.py -c hyperparams/multistep/config_cifar10_coteach+_rn18.json -d 0  --lr_scheduler multistep   --dataset cifar10 --traintools coteaching --no_wandb  --dynamic --dataseed 123 --every 10 --warmup 40
