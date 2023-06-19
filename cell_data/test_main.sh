#!/bin/bash
echo $d $h env
source /projectnb/ivc-ml/siqiwang/anaconda3/etc/profile.d/conda.sh
conda activate env
env|grep -i cuda
#$ -l h_rt=20:00:00
#$ -pe omp 8
#$ -l mem_per_core=12G
#$ -l gpus=2
#$ -P morphem

BATCH_SIZE=128
TAG="crust_w_cdrp_lr-1"
EVAL_DATASET='cdrp'
NUM_PROCESSORS=12
DATASET='cdrp'
python cell_data/simple_multi_main.py --num_gpus 4 --num_processors ${NUM_PROCESSORS} --net 'full_effb0' \
    --dataset ${DATASET} --eval_dataset ${EVAL_DATASET} --name ${TAG} --batch-size ${BATCH_SIZE}\
    --max_images_per_treatment 150 --dim_embed 672 --lr 1e-1 --epochs 50 --use_crust_k --fl-ratio 0.6\
    --start_epoch 10 --max_num_treatment 100 --predict-batch-size 32
    
conda deactivate
