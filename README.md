# KADGN
This repository provides PyTorch implementations of **KADGN** as described in the paper: **Knowledge Augmented Dual-attention Gating Network in Knowledge Graph for Link Prediction**.


![framework](https://github.com/22zwChen/KADGN/blob/07ffcc09b47808bce4d3c606e7680c8c4a554e5e/framework.png)

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
Now you are ready to train and evaluate KADGN. To reproduce the results provided in the paper, please execute the corresponding command for each dataset as follows:

#### FB15k-237
    python train.py --data FB15k-237 --epoch 1000 --batch 128 --gcn_drop 0.6 --embed_dim 300 --num_heads 3

#### WN18RR
    python train.py --data WN18RR --epoch 800 --batch 128 --gcn_drop 0.5 --embed_dim 200 --num_heads 1
    
#### Kinship
    python train.py --data kinship --epoch 1000 --batch 128 --gcn_drop 0.4 --embed_dim 300 --num_heads 3

#### Spotify
    python train.py --data spotify --epoch 1000 --batch 512 --gcn_drop 0.5 --embed_dim 200 --num_heads 3

#### Credit
    python train.py --data credit --epoch 1000 --batch 128 --gcn_drop 0.5 --embed_dim 300 --num_heads 3

### Ablation

#### -gate
Change the code in the top of **./model/BAGCN.py**:  
    agg = "gate2" -> agg = "MLP"

Then run the command mentioned above.

#### single gate
Change the code in the top of **./model/BAGCN.py**:  
    agg = "gate2" -> agg = "gate1" 

Then run the command mentioned above.

#### -inv
Before you do the ablation experiment (**-inv**), you need to replace these files as below and remember to backup original files:  
    ./helper.py -> ./helper_abl.py  
    ./model/BAGCN.py -> ./model/BAGCN_abl.py  

Then run the code of ablation experiments by another python file:  
    ./train_ablation.py


## Acknowledgement
We refer to the code of [D-AEN](https://github.com/hcfun/D-AEN) and [RAKGE](https://github.com/learndatalab/RAKGE). Thanks for their contributions.
