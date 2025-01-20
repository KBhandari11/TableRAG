from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
import os
import csv
import json
import argparse

def linearize_table(file_path):
    retriever_header = []
    retriever_sequences = []
    reader_sequences = []

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        # The first row is the header
        headers = rows[0]  
        data_rows = rows[1:]

        # Clean headers and data rows
        headers = [header.replace("\n", " ") for header in headers]
        data_rows = [[cell.replace("\n", " ") for cell in row] for row in data_rows]

        # Linearize for the retriever
        retriever_seq = "<SOT> [table title] <EOT> <BOC> " + " <SOC> ".join(headers) + " <EOC>"
        retriever_header = retriever_seq
        for row in data_rows:
            retriever_seq += " <BOR>" + " <SOR> ".join(row) + " <EOR>"
        retriever_sequences.append(retriever_seq)

        # # Linearize for the reader
        # reader_seq = "[HEAD] " + " | ".join(headers)
        # for row_id, row in enumerate(data_rows, start=1):
        #     reader_seq += f" [ROW] {row_id} : " + " | ".join(row)
        # reader_sequences.append(reader_seq)

    return retriever_header, retriever_sequences[0]


def create_queries_file(output_path):
    tsv_input_path = "/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/data/pristine-unseen-tables.tsv"

    with open(tsv_input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        reader = csv.DictReader(infile, delimiter='\t')
        id_counter = 0

        for row in reader:
            question = row['utterance']
            writer.writerow([id_counter, question])
            id_counter += 1


def merge_tables(input_dir, output_file, only_headers, csv_folder, table_id_to_path, current_id):
    table_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    total_tables = 0
    # print("TableID to path variable is of type ",type(table_id_to_path))
    with open(output_file, 'a', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter='\t')

        for table_file in table_files:
            table_path = os.path.join(input_dir, table_file)
            retriever_header, retriever_linearized = linearize_table(table_path)

            if only_headers:
                writer.writerow([current_id, retriever_header])
            else:
                writer.writerow([current_id, retriever_linearized])

            table_id_to_path.append({current_id: f"csv/{csv_folder}/{table_file}"})
            current_id += 1
            total_tables = current_id

    return total_tables


def check_and_delete_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
        except Exception as e:
            print(f"Error deleting file: {e}")
    else:
        print(f"File '{file_path}' does not exist.")


def preprocess_tables(only_headers, index_file_path):
    table_id_to_path = []
    table_id_to_path_file = '/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/tableId_to_path.json'
    check_and_delete_file(table_id_to_path_file)
    check_and_delete_file(index_file_path)
    table_id = 0
    for i in range(5):
        directory = f"/scratch/asing725/IP/Multi-Table-RAG/WikiTableQuestions/csv/20{i}-csv"
        table_id = merge_tables(directory, index_file_path, only_headers, f"20{i}-csv", table_id_to_path, table_id)
        # print(table_id_to_path)
        print(f"Processed directory 20{i}-csv")

    with open(table_id_to_path_file, 'w') as file:
        json.dump(table_id_to_path, file, indent=4)


def index_tables(gpu_count, index_name, only_headers):
    index_path = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/"
    index_path += "wtq_table_headers_only_index.tsv" if only_headers else "wtq_full_table_index.tsv"

    checkpoint_path = "/scratch/asing725/IP/Multi-Table-RAG/ColBERT_pretrained/colbertv2.0"

    with Run().context(RunConfig(nranks=gpu_count, experiment="wtq-test")):
        config = ColBERTConfig(nbits=2, root=checkpoint_path)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=index_path, overwrite=True)

def retrieve_table(gpu_count, root_dir, index_name, queries_file, top_k, output_file):
    with Run().context(RunConfig(nranks=gpu_count, experiment="wtq-test")):
        config = ColBERTConfig(root=root_dir)
        searcher = Searcher(index=index_name, config=config)
        queries = Queries(queries_file)
        ranking_results = searcher.search_all(queries, top_k)
        ranking_results.save(output_file)
    print(f"Raw Rankings saved to {output_file}")

# Generates index and a ranking file 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Index WikiTableQuestions dataset with ColBERT.")
    parser.add_argument("--k", type=int, required=True, help="Number of top results to retrieve for each query.")
    parser.add_argument("--full_table_index", type=str, required=True, help="Path to store table id , 'complete table' mapping.[Table Id, Linearized table]")
    parser.add_argument("--only_header_index", type=str, required=True, help="Path to store table id , 'table headers' mapping.[Table Id, Table Headers]")
    parser.add_argument("--table_index_name", type=str, required=True, help="Name for the Index", default="wtq_full_table.nbits=2")
    parser.add_argument("--ranking_output_name", type=str, required=True, help="Path to save raw ranking output", default="wtq_full_table.ranking.tsv")
    parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--only_headers", type=int, default=0, help="Uses only headers for indexing if set to 1.")

    args = parser.parse_args()

    index_file_path = args.only_header_index if args.only_headers else args.full_table_index
    preprocess_tables(args.only_headers, index_file_path)
    index_tables(args.gpus, args.table_index_name, args.only_headers)

    queries_path = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/data/wtq_question_index.tsv"
    create_queries_file(queries_path)

    experiments_root = "/scratch/asing725/IP/Multi-Table-RAG/TableRAG/src/ColBERT/experiments"
    retrieve_table(args.gpus, experiments_root, args.table_index_name, queries_path, args.k, args.ranking_output_name)
