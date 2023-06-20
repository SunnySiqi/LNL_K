# LNL_K
Implementation of the paper "LNL+K: Learning with Noisy Labels and Noise Source Distribution Knowledge"

## Requirements
The code has been written using Python3 (3.10.4), run `pip install -r requirements.txt` to install relevant python packages.

## Training
### CIFAR dataset with synthesized noise.
Code has been modified from the original FINE implementation: https://github.com/Kthyeon/FINE_official/. \
Please find the adaptation methods implementation in 'synthesized_data/trainer' folder. 
+ CRUST/CRUST+k: 'crust_trainer.py'
+ FINE/FINE+K: 'dynamic_trainer.py'
+ SFT/SFT+k: 'sft_trainer.py'.
#### Arguments settings & running experiments
Config files are in 'synthesized_data/hyperparams/multistep'.\
Run bash scripts in 'synthesized_data/scripts/sample_selection_based'


### Cell dataset BBBC036 with natural noise.
BBBC036 dataset is available at https://bbbc.broadinstitute.org/BBBC036. \
Please find the adaptation methods implementation in 'adaptation_methods' folder. 
+ CRUST/CRUST+k: 'crust_k.py'
+ FINE/FINE+K: 'dynamic_k.py'
+ SFT/SFT+k: 'sft_k.py'.
#### Arguments settings & running experiments
Arguments are in 'cell_data/simple_multi_main.py'. \
An example bash script is 'cell_data/test_main.sh'

## Reference Code
 - https://github.com/Kthyeon/FINE_official/
 - https://github.com/snap-stanford/crust/
 - https://github.com/1998v7/self-filtering
 - https://github.com/bhanML/Co-teaching
 - https://github.com/LiJunnan1992/DivideMix
 - https://github.com/shengliu66/ELR


