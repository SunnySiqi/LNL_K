#!/bin/bash

python main.py -d 1  -c hyperparams/multistep/config_cifar10_oracle_rn34.json --lr_scheduler multistep --arch rn34  --traintools robustloss --no_wandb --seed 456
