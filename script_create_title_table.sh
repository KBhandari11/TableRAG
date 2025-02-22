#!/bin/bash


#SBATCH --time=06:00:00  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhandk@rpi.edu
#SBATCH --gres=gpu:32g:6
#SBATCH -p dcs-2024



python ./utils/gen_table_description_nq.py /gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/nqtables/nq_tables_linearized.csv /gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/nqtables/nq_table_summary.csv
##python ./utils/create_linearized_nq.py /gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/nqtables/tables/tables.jsonl /gpfs/u/home/LLMG/LLMGbhnd/scratch/tableRAG_data/nqtables/nq_tables_linearized.csv 