#!/bin/bash
# SBATCH --job-name=sklearn
# SBATCH --output=sklearn
# SBATCH --gres=gpu:1
# SBATCH --mem=128G

source /local/scratch/rwan388/Anaconda3/etc/profile.d/conda.sh
conda activate base
# python -u train.py  --mlp=True --dname='mimic3' --num_labels=25 --num_nodes=7423 --num_labeled_data=12353 --rand_seed=0
# python -u train.py  --svm=True --dname='mimic3' --num_labels=25 --num_nodes=7423 --num_labeled_data=12353 --rand_seed=1
# python -u train.py  --svm=True --dname='mimic3' --num_labels=25 --num_nodes=7423 --num_labeled_data=12353 --rand_seed=2
# python -u train.py  --svm=True --dname='mimic3' --num_labels=25 --num_nodes=7423 --num_labeled_data=12353 --rand_seed=3
python -u train.py  --mlp=True --dname='mimic3' --num_labels=25 --num_nodes=7423 --num_labeled_data=12353 --rand_seed=4

conda deactivate
