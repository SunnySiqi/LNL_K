#!/bin/bash
#!/usr/bin/env python3.6
python main.py -d 0 \
-c hyperparams/multistep/config_cifar10_crust_w_rn34.json \
--lr_scheduler multistep \
--no_wandb \
--seed 456 \
--warmup 10 \
--traintools robustloss
