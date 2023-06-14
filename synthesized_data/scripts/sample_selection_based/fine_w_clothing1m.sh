#!/bin/bash

python main.py -c hyperparams/clothing1m/config_clothing1m_fine_w_rn50.json -d 0 --lr_scheduler clothing1m --arch rn50 --loss_fn cce  --no_wandb --traintools trainingclothing1m --distill_mode fine-gmm --warmup 5 --every 1
