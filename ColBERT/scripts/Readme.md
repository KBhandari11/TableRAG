# Official Implementation of Multi-Table RAG
This is the official implementation of Multi-Table RAG
## Retriever Model

### Code

```
module load cuda-12.1.1-gcc-12.1.0
module load gcc-12.1.0-gcc-11.2.0
mamba activate colbert
cd /scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT
clear


Full Table , K = 1
python -m scripts.main --k 1 --table_index_name "wtq_full_table.nbits=2" --ranking_output_name "wtq_full_table_k_1.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"

python -m scripts.main --k 5 --table_index_name "wtq_full_table.nbits=2" --ranking_output_name "wtq_full_table_k_5.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"

python -m scripts.main --k 10 --table_index_name "wtq_full_table.nbits=2" --ranking_output_name "wtq_full_table_k_10.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"

python -m scripts.main --k 30 --table_index_name "wtq_full_table.nbits=2" --ranking_output_name "wtq_full_table_k_30.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"

python -m scripts.main --k 50 --table_index_name "wtq_full_table.nbits=2" --ranking_output_name "wtq_full_table_k_50.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"

python -m scripts.main --k 100 --table_index_name "wtq_full_table.nbits=2" --ranking_output_name "wtq_full_table_k_100.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv"


Only Headers, K = 1
python -m scripts.main --k 1 --table_index_name "wtq_only_headers_table.nbits=2" --ranking_output_name "wtq_only_headers_table_k_1.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv" --only_headers 1

python -m scripts.main --k 5 --table_index_name "wtq_only_headers_table.nbits=2" --ranking_output_name "wtq_only_headers_table_k_5.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv" --only_headers 1

python -m scripts.main --k 10 --table_index_name "wtq_only_headers_table.nbits=2" --ranking_output_name "wtq_only_headers_table_k_10.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv" --only_headers 1

python -m scripts.main --k 30 --table_index_name "wtq_only_headers_table.nbits=2" --ranking_output_name "wtq_only_headers_table_k_30.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv" --only_headers 1

python -m scripts.main --k 50 --table_index_name "wtq_only_headers_table.nbits=2" --ranking_output_name "wtq_only_headers_table_k_50.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv" --only_headers 1

python -m scripts.main --k 100 --table_index_name "wtq_only_headers_table.nbits=2" --ranking_output_name "wtq_only_headers_table_k_100.ranking.tsv" --gpus 1 --full_table_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_full_table_index.tsv" --only_header_index "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_table_headers_only_index.tsv" --only_headers 1

For Getting Results:
python scripts/retrievalData.py 
```
