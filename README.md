# KAAGE
This repository provides PyTorch implementations of **KAAGE** as described in the paper: **Knowledge Augmented Attention Gating Embedding for Link Prediction**.


![framework](https://github.com/22zwChen/KAAGE/blob/b6d314537dca3663f5febbaa42ddb8c0bb0354d0/framework_meeting.png)

## Experiment Environment
- python 3.7.13
- torch 1.12.1+cu113
- scipy 1.7.3
- wheel 0.38.4
- numpy 1.21.6



## Basic Usage

### preparations
Since github can't create empty folders, you need to create some folders under the current path before running the code and execute the command as follows:

    mkdir log ranks ranks_index_top10 results results_index_top10 torch_saved

### Reproduce the results
Now you are ready to train and evaluate KAAGE. To reproduce the results provided in the paper, please execute the corresponding command for each dataset as follows:

#### FB15k-237
    python train.py --data FB15k-237 --epoch 1000 --batch 128 --gcn_drop 0.6 --embed_dim 300 --num_heads 3

#### WN18RR
    python train.py --data WN18RR --epoch 800 --batch 128 --gcn_drop 0.5 --embed_dim 200 --num_heads 1
    
#### Kinship
    python train.py --data kinship --epoch 1000 --batch 128 --gcn_drop 0.4 --embed_dim 300 --num_heads 3

