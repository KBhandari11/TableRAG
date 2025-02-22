#!/bin/bash


#SBATCH --time=06:00:00  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhandk@rpi.edu
#SBATCH --gres=gpu:32g:4
#SBATCH -p dcs-2024


python ./wtq_simple_embedding.py $1
#python wtq_table_rag.py $1

#qspython utils/create_retrieval_dataset_$1.py
#./totto_simple_embedding.py $1