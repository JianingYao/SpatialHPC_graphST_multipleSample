#!/bin/bash

srun --partition=gpu --mem=50G --gres=gpu:1 --pty /bin/bash

conda activate GraphST_hpc

export LD_LIBRARY_PATH=~/.conda/envs/GraphST_hpc/lib/R/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/jhpce/shared/jhpce/core/R/4.0.3/lib64/R/lib:$LD_LIBRARY_PATH

python run-graphST-batch.py













