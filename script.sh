#CUDA_VISIBLE_DEVICES=1,2,3 python test_simple_embedding.py

CUDA_VISIBLE_DEVICES=$1 python test_simple_embedding.py > result.txt 2>&1