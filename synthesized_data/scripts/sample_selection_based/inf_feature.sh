#!/bin/bash

python evaluate.py  -c hyperparams/multistep/config_cifar10_crust_rn34.json\
        -d 1\
	-r saved/models/cifar10/resnet34/MultiStepLR/CrossEntropyLoss/asym/adptive_crust/90/model_best123.pth
