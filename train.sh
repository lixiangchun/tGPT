#!/bin/bash

export OMP_NUM_THREADS=1

export TOKENIZERS_PARALLELISM=false
/opt/install/miniconda3/bin/python train.py --train-file trn.txt --val-file val.txt \
-j 1 --max-len 64 -b 8 --epochs 10 --device cpu \
--lr 0.003 --lr_scheduler_type cosine \
--output-dir ./ \
--tokenizer_dir gpt2tokenizer \
--print-freq 2


## An example for parallel training on A100
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
#-j 1 --max-len 64 \
#--lr 0.003 --lr_scheduler_type cosine \
#--output-dir ./ \
#--tokenizer_dir gpt2tokenizer \
#--print-freq 100



